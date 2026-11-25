#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/hotPages.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
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
// Pages the guest rewrites every frame stay write-watched. APS5_KYTY_STICKY_PAGES=1 (diagnostic)
// leaves them unwatched and uploads them once per submission (see ForEachModifiedRange): that
// trusts every CPU write a submission reads to land before it is submitted, which one title
// breaks (stale bytes reach the GPU), and it was slower there (15.5 vs 17.6 fps).
inline std::atomic<bool>     g_sticky_pages {std::getenv("APS5_KYTY_STICKY_PAGES") != nullptr};
// Frames the guest has submitted for display so far.
inline std::atomic<uint64_t> g_tracker_frame {1};

// AnyPS5 diagnostics (APS5_KYTY_MEMSTATS): bytes of rewritten ("hot") pages and of other dirty
// pages handed to uploads.
inline std::atomic<uint64_t> g_hot_upload_bytes {0};
inline std::atomic<uint64_t> g_cold_upload_bytes {0};
// APS5_KYTY_HOTCHECK: uploaded bytes unchanged / changed since their previous upload.
inline std::atomic<uint64_t> g_upload_same_bytes {0};
inline std::atomic<uint64_t> g_gpumod_calls {0}, g_gpumod_regions {0}, g_cpumod_calls {0}, g_cpumod_regions {0};
inline std::atomic<uint64_t> g_upload_changed_bytes {0};

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
// Exact hot pages (hotPages.h, APS5_KYTY_X_CUT letter e): set by the buffer cache, which gives the
// shadows their reader view; never with the rewritten-page mode, whose pages they replace.
inline std::atomic<bool> g_exact_hot_pages {false};
inline bool              ExactHotPagesEnabled() noexcept {
	return g_exact_hot_pages.load(std::memory_order_relaxed) && !StickyPagesEnabled();
}

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr, HotPages* hot_pages = nullptr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr), m_hot_pages(hot_pages) {
		m_cpu_dirty.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	// Grows after every change of the tracking state, made under the lock. Read without the
	// lock: an equal value means nothing changed in between (see MemoryTracker::ForEachUploadRange).
	[[nodiscard]] uint64_t Generation() const noexcept {
		return m_generation.load(std::memory_order_acquire);
	}

	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(offset, size);
		if (!GetBits<source>().Any(start, end)) {
			return false;
		}
		if constexpr (source == DirtySource::Cpu) {
			// An exact hot page keeps its dirty bit: it is modified when its bytes differ from
			// the ones its buffer got (GPU thread, like every caller of this query).
			if (HasExactHot(start, end)) {
				for (const auto [first, last]: RegionBits(m_cpu_dirty, start, end)) {
					for (auto page = first; page < last; page++) {
						if (!m_exact_hot.Get(page) || HotChanged(page)) {
							return true;
						}
					}
				}
				return false;
			}
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

	// True when the region keeps some page unwatched: its CPU state then depends on the epoch.
	[[nodiscard]] bool HasHot() const noexcept { return m_has_hot; }

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
		// Hot pages change below even when no dirty bit does.
		bool had_hot = m_has_hot;
		if constexpr (source == DirtySource::Cpu && enable) {
			if (m_gpu_dirty.Any(start, end)) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
			// Marked CPU-modified from outside (a fault through another cache's watch, a buffer
			// dropped, guest memory adopted): the buffer no longer surely holds the shadow's
			// bytes. The page becomes an ordinary CPU-dirty page, unwatched as it already is.
			had_hot |= DemoteExactHot(start, end);
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
			// AnyPS5: only when the two writes come in the same or consecutive frames. A page
			// the guest writes every few frames was kept unwatched too and uploaded every frame
			// unchanged (83% of 20 MiB a frame in one title).
			if (StickyPagesEnabled()) {
				const auto frame = TrackerFrame();
				if (frame != m_rewritten_frame) {
					m_rewritten[frame & 1].Clear();
					if (frame != m_rewritten_frame + 1) {
						m_rewritten[(frame + 1) & 1].Clear();
					}
					m_rewritten_frame = frame;
				}
				auto&       current  = m_rewritten[frame & 1];
				const auto& previous = m_rewritten[(frame + 1) & 1];
				for (const auto [first, last]: changed) {
					for (auto page = first; page < last; page++) {
						if (current.Get(page) || previous.Get(page)) {
							if (!m_has_hot) {
								m_has_hot   = true;
								m_hot_since = frame;
							}
							m_hot.Set(page);
						} else {
							current.Set(page);
						}
					}
				}
			}
		}
		if constexpr (source == DirtySource::Cpu && enable) {
			NoteWriteCycles(changed);
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
		// A call that changes nothing (a buffer written by every draw, GPU-dirty already) keeps
		// the generation, so upload memos of the region's other ranges stay valid.
		if (had_hot || !changed.None()) {
			Bump();
		}
	}

	// promote: whether pages rewritten every frame may become exact hot pages here (not when
	// the range is about to be marked GPU-modified).
	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t offset, uint64_t size, Func&& func, bool promote = true) {
		(void)promote; // CPU uploads only
		const auto [start, end] = GetPageRange(offset, size);
		auto&      bits         = GetBits<source>();
		if (!bits.Any(start, end)) {
			return;
		}
		if constexpr (clear && source == DirtySource::Cpu) {
			if (ExactHotActive()) {
				// Reported: dirty pages, but exact hot pages only when their bytes changed.
				// Watched again and cleared: dirty pages that are not (and do not become) hot.
				RegionBits upload(bits, start, end);
				RegionBits watch(bits, start, end);
				const bool promoted = SplitExactHot(upload, watch, promote);
				for (const auto [first, last]: watch) {
					bits.UnsetRange(first, last);
				}
				UpdateProtection<true, false>(watch);
				// Hot pages that only got compared change no state.
				if (promoted || !watch.None()) {
					Bump();
				}
				for (const auto [first, last]: upload) {
					func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
				}
				return;
			}
		}
		if constexpr (clear && source == DirtySource::Cpu) {
			if (m_has_hot && StickyPagesEnabled()) {
				const auto epoch = TrackerEpoch();
				if (TrackerFrame() - m_hot_since > HotPageLifetime()) {
					// Pages stop counting as rewritten after a while; they qualify again by faulting.
					m_hot.Clear();
					m_rewritten[0].Clear();
					m_rewritten[1].Clear();
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
					Bump();
					for (const auto [first, last]: upload) {
						if (Common::DebugCounters::g_aps5_stats) {
							uint64_t hot = 0;
							for (auto page = first; page < last; page++) {
								hot += m_hot.Get(page) ? 1 : 0;
							}
							g_hot_upload_bytes.fetch_add(hot * TRACKER_PAGE_SIZE, std::memory_order_relaxed);
							g_cold_upload_bytes.fetch_add((last - first - hot) * TRACKER_PAGE_SIZE,
							                              std::memory_order_relaxed);
						}
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
			Bump();
		}
		for (const auto [first, last]: mask) {
			if constexpr (source == DirtySource::Cpu) {
				if (Common::DebugCounters::g_aps5_stats) {
					g_cold_upload_bytes.fetch_add((last - first) * TRACKER_PAGE_SIZE, std::memory_order_relaxed);
				}
			}
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	// ForEachModifiedRange<DirtySource::Cpu, true> for `count` ranges of this region, range_at(i)
	// giving the i-th as {offset, size}, with one protection change for all of them: pages left
	// between the ranges that are watched already join the same protection call instead of
	// splitting it. func(i, address, bytes) gets the pages each call would have reported, in the
	// same order. Only while StickyPagesEnabled() is false (hot pages take the other path).
	template <typename RangeAt, typename Func>
	void ForEachCpuModifiedRangeBatch(size_t count, RangeAt&& range_at, Func&& func) {
		RegionBits mask;
		bool       any = false;
		for (size_t index = 0; index < count; index++) {
			const auto [offset, size] = range_at(index);
			const auto [start, end]   = GetPageRange(offset, size);
			if (!m_cpu_dirty.Any(start, end)) {
				continue;
			}
			for (const auto [first, last]: RegionBits(m_cpu_dirty, start, end)) {
				mask.SetRange(first, last);
			}
			any = true;
		}
		if (!any) {
			return;
		}
		// As ForEachModifiedRange: exact hot pages stay dirty and unwatched, reported only when
		// their bytes changed; the other dirty pages are cleared and watched.
		RegionBits watch(mask);
		const bool promoted = ExactHotActive() && SplitExactHot(mask, watch, true);
		for (const auto [first, last]: watch) {
			m_cpu_dirty.UnsetRange(first, last);
		}
		UpdateProtection<true, false>(watch);
		// Hot pages that only got compared change no state.
		if (promoted || !watch.None()) {
			Bump();
		}
		for (size_t index = 0; index < count; index++) {
			const auto [offset, size] = range_at(index);
			const auto [start, end]   = GetPageRange(offset, size);
			// A page two ranges share is reported once, for the first, as separate calls would.
			RegionBits own(mask, start, end);
			mask.UnsetRange(start, end);
			for (const auto [first, last]: own) {
				func(index, m_cpu_addr + first * TRACKER_PAGE_SIZE,
				     (last - first) * TRACKER_PAGE_SIZE);
			}
		}
	}

	// ChangeState<DirtySource::Gpu, false> for `count` ranges of this region, range_at(i) giving
	// the i-th as {offset, size}, with one protection change for all of them: pages between the
	// ranges whose protection already is what the cleared pages get join the same call instead of
	// splitting it. Every page ends with the watchers and protection the separate calls leave.
	template <typename RangeAt>
	void ClearGpuModifiedBatch(size_t count, RangeAt&& range_at) {
		RegionBits changed;
		for (size_t index = 0; index < count; index++) {
			const auto [offset, size] = range_at(index);
			const auto [start, end]   = GetPageRange(offset, size);
			if (!m_gpu_dirty.Any(start, end)) {
				continue;
			}
			for (const auto [first, last]: RegionBits(m_gpu_dirty, start, end)) {
				changed.SetRange(first, last);
			}
			m_gpu_dirty.UnsetRange(start, end);
		}
		UpdateProtection<false, true>(changed);
		// As ChangeState: a region with hot pages bumps even when nothing changed.
		if (m_has_hot || !changed.None()) {
			Bump();
		}
	}

	// Exact hot pages among pages [start, end) of the region; lock-free like IsModified.
	[[nodiscard]] bool HasExactHot(size_t start, size_t end) const noexcept {
		return m_exact_count.load(std::memory_order_acquire) != 0 && m_exact_hot.Any(start, end);
	}
	[[nodiscard]] bool HasExactHotBytes(uint64_t offset, uint64_t size) const noexcept {
		const auto [start, end] = GetPageRange(offset, size);
		return HasExactHot(start, end);
	}

	// GPU thread: whether the exact hot page at `offset` differs from its upload; false for a
	// page that is not hot.
	[[nodiscard]] bool ExactHotChangedAt(uint64_t offset) const {
		const auto page = static_cast<size_t>(offset / TRACKER_PAGE_SIZE);
		return m_exact_hot.Get(page) && HotChanged(page);
	}

	// GPU thread: the shadow slot of the page at `offset`, HotPages::NO_SLOT when it is not
	// an exact hot page. Lock-free: a page demoted meanwhile may still report its old slot, which
	// is not reused before the next tracker upload (see HotPages).
	[[nodiscard]] uint16_t ExactHotSlot(uint64_t offset) const noexcept {
		const auto* history = m_history.load(std::memory_order_acquire);
		if (history == nullptr || m_exact_count.load(std::memory_order_acquire) == 0) {
			return HotPages::NO_SLOT;
		}
		return history->slot[static_cast<size_t>(offset / TRACKER_PAGE_SIZE)].load(
		    std::memory_order_acquire);
	}

	// Under the lock: exact hot pages of the range become ordinary CPU-dirty pages (dirty and
	// unwatched, as they are). The next upload of each watches it first and copies it after, as
	// for any page a fault reported. True when there were any.
	bool DemoteExactHot(size_t start, size_t end) {
		if (!HasExactHot(start, end)) {
			return false;
		}
		auto* history = m_history.load(std::memory_order_relaxed);
		for (const auto [first, last]: RegionBits(m_exact_hot, start, end)) {
			for (auto page = first; page < last; page++) {
				const auto slot =
				    history->slot[page].exchange(HotPages::NO_SLOT, std::memory_order_acq_rel);
				if (slot != HotPages::NO_SLOT) {
					m_hot_pages->Release(slot);
				}
				// It qualifies again only by faulting in frames in a row.
				history->streak[page] = 0;
			}
		}
		m_exact_hot.UnsetRange(start, end);
		m_exact_count.store(m_exact_hot.None() ? 0u : 1u, std::memory_order_release);
		Bump();
		return true;
	}
	bool DemoteExactHotBytes(uint64_t offset, uint64_t size) {
		const auto [start, end] = GetPageRange(offset, size);
		return DemoteExactHot(start, end);
	}

	TrackingSpinLock lock;

private:
	void Bump() noexcept { m_generation.fetch_add(1, std::memory_order_release); }

	// Per page of a region whose pages the CPU writes: when and how often it went through a
	// write fault, and its exact hot page slot. Created on the first such fault (under the lock)
	// and kept, so lock-free readers may hold it.
	struct PageHistory {
		std::array<uint16_t, TRACKER_REGION_PAGES>              frame {};  // low bits of the last fault's frame
		std::array<uint8_t, TRACKER_REGION_PAGES>               streak {}; // frames in a row with a fault
		std::array<std::atomic<uint16_t>, TRACKER_REGION_PAGES> slot;
		PageHistory() {
			for (auto& entry: slot) {
				entry.store(HotPages::NO_SLOT, std::memory_order_relaxed);
			}
		}
	};
	// Frames in a row a page must take a write fault before it is kept hot.
	static constexpr uint8_t HOT_STREAK = 3;

	[[nodiscard]] bool ExactHotActive() const noexcept {
		return m_hot_pages != nullptr && ExactHotPagesEnabled() &&
		       (m_exact_count.load(std::memory_order_relaxed) != 0 ||
		        m_history.load(std::memory_order_relaxed) != nullptr);
	}

	[[nodiscard]] bool HotChanged(size_t page) const {
		const auto slot =
		    m_history.load(std::memory_order_acquire)->slot[page].load(std::memory_order_acquire);
		// Demoted meanwhile: an ordinary dirty page.
		return slot == HotPages::NO_SLOT ||
		       m_hot_pages->Changed(slot, m_cpu_addr + page * TRACKER_PAGE_SIZE, TrackerFrame());
	}

	// Under the lock: counts the frames in a row in which each page of `changed` (pages a CPU
	// write made dirty) took a write fault.
	void NoteWriteCycles(const RegionBits& changed) {
		if (m_hot_pages == nullptr || !ExactHotPagesEnabled() || changed.None()) {
			return;
		}
		auto* history = m_history.load(std::memory_order_relaxed);
		if (history == nullptr) {
			m_history_storage = std::make_unique<PageHistory>();
			history           = m_history_storage.get();
			m_history.store(history, std::memory_order_release);
		}
		const auto frame = static_cast<uint16_t>(TrackerFrame());
		for (const auto [first, last]: changed) {
			for (auto page = first; page < last; page++) {
				auto& seen   = history->frame[page];
				auto& streak = history->streak[page];
				if (seen == frame && streak != 0) {
					continue;
				}
				streak = (streak != 0 && seen == static_cast<uint16_t>(frame - 1u))
				             ? static_cast<uint8_t>(std::min<int>(streak + 1, UINT8_MAX))
				             : uint8_t {1};
				seen   = frame;
			}
		}
	}

	// Under the lock, GPU thread, in an upload: `upload` and `watch` start as the dirty pages
	// of the call. Exact hot pages leave `watch` (they stay dirty and unwatched) and leave
	// `upload` when their bytes equal their shadow. With `promote`, a dirty page that took a
	// write fault in each of the last HOT_STREAK frames becomes hot: it leaves `watch` and is
	// uploaded, which fills its shadow (BufferCache::UploadCopies). True when a page became hot.
	bool SplitExactHot(RegionBits& upload, RegionBits& watch, bool promote) {
		bool promoted = false;
		auto*      history = m_history.load(std::memory_order_relaxed);
		const auto frame   = TrackerFrame();
		promote            = promote && history != nullptr && m_hot_pages->PromotionAllowed(frame);
		const RegionBits dirty(upload);
		for (const auto [first, last]: dirty) {
			for (auto page = first; page < last; page++) {
				if (m_exact_hot.Get(page)) {
					watch.Unset(page);
					if (!HotChanged(page)) {
						upload.Unset(page);
					}
					continue;
				}
				if (!promote || history->streak[page] < HOT_STREAK ||
				    static_cast<uint16_t>(static_cast<uint16_t>(frame) - history->frame[page]) > 1u) {
					continue;
				}
				const auto address = m_cpu_addr + page * TRACKER_PAGE_SIZE;
				if (m_hot_pages->Read(address) == nullptr) {
					continue;
				}
				const auto slot = m_hot_pages->Acquire(address, frame);
				if (slot == HotPages::NO_SLOT) {
					promote = false;
					continue;
				}
				history->slot[page].store(slot, std::memory_order_release);
				m_exact_hot.Set(page);
				m_exact_count.store(1u, std::memory_order_release);
				watch.Unset(page);
				promoted = true;
			}
		}
		return promoted;
	}

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
	RegionBits   m_rewritten[2]; // write faults of the frame (index: frame parity) and the one before
	uint64_t     m_rewritten_frame = 0;
	RegionBits   m_hot;       // kept CPU-dirty and unwatched
	RegionBits   m_hot_fresh; // hot pages uploaded in submission m_fresh_epoch
	uint64_t     m_fresh_epoch = 0;
	uint64_t     m_hot_since   = 0;
	bool         m_has_hot     = false;
	// Exact hot pages (hotPages.h): dirty, unwatched, compared with their shadow when queried.
	HotPages*                    m_hot_pages = nullptr;
	RegionBits                   m_exact_hot;
	std::atomic<uint32_t>        m_exact_count {0}; // nonzero while m_exact_hot has a page
	std::unique_ptr<PageHistory> m_history_storage;
	std::atomic<PageHistory*>    m_history {nullptr};
	// Starts at 1, so a sum over existing regions is never 0.
	std::atomic<uint64_t> m_generation {1};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
