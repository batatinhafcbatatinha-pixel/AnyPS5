#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
			// Wait on reads, so the holder's cache line is not written until the lock is free.
			while (m_lock.test(std::memory_order_relaxed)) {
#if defined(__x86_64__) || defined(_M_X64)
				_mm_pause();
#elif defined(_M_ARM64)
				__yield();
#elif defined(__aarch64__)
				asm volatile("yield");
#endif
			}
		}
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}

private:
	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return GetCurrentThreadId();
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		static thread_local const uint32_t tid =
		    static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
		return tid;
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

// Guest command submissions processed so far: CPU writes to memory a submission reads land
// before it is submitted, so a page uploaded during one submission stays current for it.
inline std::atomic<uint64_t> g_tracker_epoch {1};
inline std::atomic<bool>     g_sticky_pages {true};
// Frames the guest has submitted for display so far.
inline std::atomic<uint64_t> g_tracker_frame {1};

inline uint64_t TrackerEpoch() noexcept {
	return g_tracker_epoch.load(std::memory_order_relaxed);
}
inline void AdvanceTrackerEpoch() noexcept {
	g_tracker_epoch.fetch_add(1, std::memory_order_relaxed);
}
inline uint64_t TrackerFrame() noexcept {
	return g_tracker_frame.load(std::memory_order_relaxed);
}
inline void AdvanceTrackerFrame() noexcept {
	g_tracker_frame.fetch_add(1, std::memory_order_relaxed);
}
inline bool StickyPagesEnabled() noexcept {
	return g_sticky_pages.load(std::memory_order_relaxed);
}

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		m_cpu_dirty.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(offset, size);
		if (!GetBits<source>().Any(start, end)) {
			return false;
		}
		if constexpr (source == DirtySource::Cpu) {
			// Rewritten pages uploaded in this submission are current in their buffers.
			if (m_has_hot && StickyPagesEnabled() && m_fresh_epoch == TrackerEpoch()) {
				for (const auto [first, last]: RegionBits(m_cpu_dirty, start, end)) {
					for (auto page = first; page < last; page++) {
						if (!m_hot.Get(page) || !m_hot_fresh.Get(page)) {
							return true;
						}
					}
				}
				return false;
			}
		}
		return true;
	}

	// True when a page of the range is kept unwatched (see ChangeState): its writes raise no fault.
	[[nodiscard]] bool IsHot(uint64_t offset, uint64_t size) const {
		if (!m_has_hot) {
			return false;
		}
		const auto [start, end] = GetPageRange(offset, size);
		return m_hot.Any(start, end);
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t offset, uint64_t size) {
		const auto [start, end] = GetPageRange(offset, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (m_gpu_dirty.Any(start, end)) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			// GPU-written pages leave the rewritten set: they are write-watched again.
			if (m_has_hot) {
				RegionBits watch(m_cpu_dirty, start, end);
				for (const auto [first, last]: RegionBits(m_cpu_dirty, start, end)) {
					for (auto page = first; page < last; page++) {
						if (!m_hot.Get(page)) {
							watch.Unset(page);
						}
					}
				}
				m_hot.UnsetRange(start, end);
				for (const auto [first, last]: watch) {
					m_cpu_dirty.UnsetRange(first, last);
				}
				UpdateProtection<true, false>(watch);
			}
		}
		auto&      bits = GetBits<source>();
		RegionBits changed = enable ? RegionBits(~bits, start, end) : RegionBits(bits, start, end);
		if constexpr (source == DirtySource::Cpu && enable) {
			// A page written again after its upload is rewritten by the guest every frame:
			// keeping it unwatched and uploading it once per submission saves a fault and two
			// protection changes per write.
			if (StickyPagesEnabled()) {
				for (const auto [first, last]: changed) {
					for (auto page = first; page < last; page++) {
						if (m_rewritten.Get(page)) {
							if (!m_has_hot) {
								m_has_hot   = true;
								m_hot_since = TrackerFrame();
							}
							m_hot.Set(page);
						} else {
							m_rewritten.Set(page);
						}
					}
				}
			}
		}
		if constexpr (enable) {
			bits.SetRange(start, end);
		} else {
			bits.UnsetRange(start, end);
		}
		if constexpr (source == DirtySource::Cpu) {
			UpdateProtection<!enable, false>(changed);
		} else {
			UpdateProtection<enable, true>(changed);
		}
	}

	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t offset, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(offset, size);
		auto&      bits         = GetBits<source>();
		if (!bits.Any(start, end)) {
			return;
		}
		if constexpr (clear && source == DirtySource::Cpu) {
			if (m_has_hot && StickyPagesEnabled()) {
				const auto epoch = TrackerEpoch();
				if (TrackerFrame() - m_hot_since > HotPageLifetime()) {
					// Pages stop counting as rewritten after a while; they qualify again by faulting.
					m_hot.Clear();
					m_rewritten.Clear();
					m_has_hot = false;
				} else {
					if (m_fresh_epoch != epoch) {
						m_hot_fresh.Clear();
						m_fresh_epoch = epoch;
					}
					RegionBits upload(bits, start, end);
					RegionBits watch(bits, start, end);
					for (const auto [first, last]: RegionBits(bits, start, end)) {
						for (auto page = first; page < last; page++) {
							if (!m_hot.Get(page)) {
								continue;
							}
							watch.Unset(page);
							if (m_hot_fresh.Get(page)) {
								upload.Unset(page);
							} else {
								m_hot_fresh.Set(page);
							}
						}
					}
					for (const auto [first, last]: watch) {
						bits.UnsetRange(first, last);
					}
					UpdateProtection<true, false>(watch);
					for (const auto [first, last]: upload) {
						func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
					}
					return;
				}
			}
		}
		RegionBits mask(bits, start, end);
		if constexpr (clear) {
			bits.UnsetRange(start, end);
			if constexpr (source == DirtySource::Cpu) {
				UpdateProtection<true, false>(mask);
			} else {
				UpdateProtection<false, true>(mask);
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	TrackingSpinLock lock;

private:
	// CPU-dirty pages are not write-watched; GPU-dirty pages are read-watched.
	template <bool track, bool is_read>
	void UpdateProtection(RegionBits& changed) {
		if (!changed.None()) {
			m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, changed);
		}
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	[[nodiscard]] static std::pair<size_t, size_t> GetPageRange(uint64_t offset, uint64_t size) {
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	// Frames a page stays in the rewritten set before it is write-watched again. A page the guest
	// stopped writing is uploaded on every submission meanwhile, so the set is rebuilt often: a
	// page written every frame pays two faults per period to stay in it.
	// KYTY_REWRITTEN_FRAMES=n sets another period.
	[[nodiscard]] static uint64_t HotPageLifetime() noexcept {
		static const uint64_t frames = [] {
			const char* value = std::getenv("KYTY_REWRITTEN_FRAMES");
			const auto  given = value != nullptr ? std::strtoull(value, nullptr, 10) : 0;
			return given != 0 ? given : uint64_t {64};
		}();
		return frames;
	}

	PageManager& m_page_manager;
	uint64_t     m_cpu_addr = 0;
	RegionBits   m_cpu_dirty;
	RegionBits   m_gpu_dirty;
	RegionBits   m_rewritten; // written again by the CPU after an upload
	RegionBits   m_hot;       // kept CPU-dirty and unwatched
	RegionBits   m_hot_fresh; // hot pages uploaded in submission m_fresh_epoch
	uint64_t     m_fresh_epoch = 0;
	uint64_t     m_hot_since   = 0;
	bool         m_has_hot     = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
