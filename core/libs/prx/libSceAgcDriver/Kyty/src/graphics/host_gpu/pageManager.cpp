#include "graphics/host_gpu/pageManager.h"
#include "common/srwLock.h"

#include "common/alignment.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace Libs::Graphics {
namespace {

constexpr uint64_t PAGE_SIZE    = TRACKER_PAGE_SIZE;
constexpr uint64_t REGION_SIZE  = TRACKER_REGION_SIZE;
constexpr uint64_t ADDRESS_SIZE = TRACKER_ADDRESS_SIZE;
constexpr uint64_t REGION_COUNT = ADDRESS_SIZE / REGION_SIZE;

constexpr uint64_t REGION_PAGES = REGION_SIZE / PAGE_SIZE;

[[noreturn]] void Fatal(const char* format, ...) {
	std::fputs("PageManager fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	std::_Exit(322);
}

class SpinGuard final {
public:
	explicit SpinGuard(std::atomic_flag& lock): m_lock(lock) {
		while (m_lock.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
			_mm_pause();
#elif defined(_M_ARM64)
			__yield();
#elif defined(__aarch64__)
			asm volatile("yield");
#endif
		}
	}
	~SpinGuard() { m_lock.clear(std::memory_order_release); }
	KYTY_CLASS_NO_COPY(SpinGuard);

private:
	std::atomic_flag& m_lock;
};

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

} // namespace

struct PageManager::Impl {
	struct PageState {
		uint8_t write_watchers  = 0;
		uint8_t access_watchers = 0;

		[[nodiscard]] Common::VirtualMemory::Mode Perms() const noexcept {
			if (access_watchers != 0) {
				return Common::VirtualMemory::Mode::NoAccess;
			}
			if (write_watchers != 0) {
				return Common::VirtualMemory::Mode::Read;
			}
			return Common::VirtualMemory::Mode::ReadWrite;
		}

		template <int delta, bool is_read>
		uint32_t AddDelta(uint64_t address) {
			static_assert(delta >= -1 && delta <= 1);
			if constexpr (is_read) {
				if constexpr (delta == 1) {
					if (access_watchers == UINT8_MAX) {
						Fatal("read-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++access_watchers;
				} else if constexpr (delta == -1) {
					if (access_watchers == 0) {
						Fatal("read-watcher underflow at 0x%016" PRIx64, address);
					}
					return --access_watchers;
				} else {
					return access_watchers;
				}
			} else {
				if constexpr (delta == 1) {
					if (write_watchers == UINT8_MAX) {
						Fatal("write-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++write_watchers;
				} else if constexpr (delta == -1) {
					if (write_watchers == 0) {
						Fatal("write-watcher underflow at 0x%016" PRIx64, address);
					}
					return --write_watchers;
				} else {
					return write_watchers;
				}
			}
		}
	};
	static_assert(sizeof(PageState) == 2);

	struct Region {
		std::atomic_flag                    lock = ATOMIC_FLAG_INIT;
		std::array<PageState, REGION_PAGES> pages;
		// Pages whose OS protection may lag their watcher counts after a deferred change, and
		// whether the region is in `deferred` (both under `lock`).
		RegionBits pending;
		bool       queued = false;
	};

	Impl() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		if (info.dwPageSize != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32,
			      static_cast<uint32_t>(info.dwPageSize));
		}
#elif defined(__APPLE__)
		// Under Rosetta the host page size is 4 KB, matching TRACKER_PAGE_SIZE.
		if (static_cast<uint64_t>(getpagesize()) != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32, static_cast<uint32_t>(getpagesize()));
		}
#else
		const auto host_page_size = ::sysconf(_SC_PAGESIZE);
		if (host_page_size < 0 || static_cast<uint64_t>(host_page_size) != PAGE_SIZE) {
			Fatal("unsupported host page size %ld", static_cast<long>(host_page_size));
		}
#endif
		regions = std::make_unique<std::atomic<Region*>[]>(REGION_COUNT);
	}

	~Impl() {
		for (const auto& region: region_storage) {
			SpinGuard lock(region->lock);
			for (auto& page: region->pages) {
				if (page.write_watchers != 0 || page.access_watchers != 0) {
					Fatal("destroyed with live page state");
				}
			}
		}
	}

	Region* FindRegion(uint64_t vaddr) const noexcept {
		return regions[TrackerOffset(vaddr) / REGION_SIZE].load(std::memory_order_acquire);
	}

	Region* GetOrCreateRegion(uint64_t vaddr) {
		const auto index = TrackerOffset(vaddr) / REGION_SIZE;
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		std::lock_guard lock(region_mutex);
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		auto  region = std::make_unique<Region>();
		auto* ptr    = region.get();
		region_storage.push_back(std::move(region));
		regions[index].store(ptr, std::memory_order_release);
		return ptr;
	}

	void Protect(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) noexcept {
		if (!Libs::LibKernel::Memory::ProtectGuestHostMemory(vaddr, size, mode)) {
			Fatal("address-space protection failed at 0x%016" PRIx64 ", mode=0x%08" PRIx32, vaddr,
			      static_cast<uint32_t>(mode));
		}
	}

	template <bool track, bool is_read, bool masked>
	void UpdateRegionWatchers(Region& region, uint64_t base_addr, size_t first, size_t last,
	                          const RegionBits* mask = nullptr, bool defer = false) {
		SpinGuard lock(region.lock);
		auto      perms                 = region.pages[first].Perms();
		uint64_t  range_begin           = 0;
		uint64_t  range_bytes           = 0;
		uint64_t  potential_range_bytes = 0;

		const auto release_pending = [&] {
			if (range_bytes != 0) {
				if (defer) {
					region.pending.SetRange(static_cast<size_t>(range_begin),
					                        static_cast<size_t>(range_begin + range_bytes / PAGE_SIZE));
					if (!region.queued) {
						region.queued = true;
						std::lock_guard deferred_lock(deferred_mutex);
						deferred.emplace_back(&region, base_addr);
						has_deferred.store(true, std::memory_order_release);
					}
				} else {
					Protect(base_addr + range_begin * PAGE_SIZE, range_bytes, perms);
				}
				range_bytes           = 0;
				potential_range_bytes = 0;
			}
		};

		constexpr int delta = track ? 1 : -1;
		for (size_t page_index = first; page_index < last; page_index++) {
			auto&      page    = region.pages[page_index];
			const auto address = base_addr + page_index * PAGE_SIZE;
			const bool update  = !masked || mask->Get(page_index);

			const auto old_perms = page.Perms();
			const auto new_count = update ? page.AddDelta<delta, is_read>(address)
			                              : page.AddDelta<0, is_read>(address);
			const auto new_perms = page.Perms();

			if (new_perms != perms) [[unlikely]] {
				release_pending();
				perms = new_perms;
			} else if (range_bytes != 0) {
				potential_range_bytes += PAGE_SIZE;
			}

			if (!update) {
				continue;
			}

			const bool watcher_edge = (track && new_count == 1) || (!track && new_count == 0);
			if (watcher_edge && old_perms != new_perms) {
				if (range_bytes == 0) {
					range_begin           = page_index;
					potential_range_bytes = PAGE_SIZE;
				}
				range_bytes = potential_range_bytes;
			}
		}

		release_pending();
	}

	void FlushDeferred() {
		std::vector<std::pair<Region*, uint64_t>> regions;
		{
			std::lock_guard deferred_lock(deferred_mutex);
			regions.swap(deferred);
			has_deferred.store(false, std::memory_order_release);
		}
		for (const auto& [region, base_addr]: regions) {
			SpinGuard lock(region->lock);
			region->queued = false;
			if (region->pending.None()) {
				continue;
			}
			// One call per run of pages with one protection that holds a pending page. Pages
			// between pending ones whose protection is the run's already have it (only pending
			// pages lag their counts), so they join the run instead of splitting it.
			const auto lo   = region->pending.FirstRange().first;
			const auto hi   = region->pending.LastRange().second;
			size_t     run  = lo;
			size_t     tail = lo; // one past the last pending page of the run
			auto       mode = region->pages[lo].Perms();
			const auto issue = [&] {
				if (tail > run) {
					Protect(base_addr + run * PAGE_SIZE, (tail - run) * PAGE_SIZE, mode);
				}
			};
			for (size_t page = lo; page < hi; page++) {
				const auto page_mode = region->pages[page].Perms();
				const bool lagging   = region->pending.Get(page);
				if (page_mode != mode) {
					issue();
					run  = page;
					tail = page;
					mode = page_mode;
				}
				if (lagging) {
					tail = page + 1;
				} else if (tail == run) {
					// No pending page in the run yet: it starts later.
					run  = page + 1;
					tail = page + 1;
				}
			}
			issue();
			region->pending.Clear();
		}
	}

	[[nodiscard]] bool HasDeferred() const noexcept {
		return has_deferred.load(std::memory_order_acquire);
	}

	template <bool track, bool is_read>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size, bool defer = false) {
		ValidateRange(vaddr, size);
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			auto*      region = track ? GetOrCreateRegion(chunk_begin) : FindRegion(chunk_begin);
			if (region == nullptr) {
				Fatal("untracking unknown page 0x%016" PRIx64, chunk_begin);
			}
			const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			UpdateRegionWatchers<track, is_read, false>(*region, region_base, first, last, nullptr,
			                                            defer);
			chunk_begin = chunk_end;
		}
	}

	void WriteGuest(uint64_t vaddr, const void* data, uint64_t size) {
		ValidateRange(vaddr, size);
		const auto* bytes = static_cast<const uint8_t*>(data);
		const auto  end   = vaddr + size;
		for (auto chunk_begin = vaddr; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			auto*      destination = reinterpret_cast<void*>(chunk_begin);
			const auto source      = bytes + (chunk_begin - vaddr);
			auto*      region      = FindRegion(chunk_begin);
			if (region == nullptr) {
				// No page here was ever watched, so none is protected by the tracker.
				std::memcpy(destination, source, chunk_end - chunk_begin);
				chunk_begin = chunk_end;
				continue;
			}
			// Watcher changes of these pages wait on the lock, so no protection change interleaves.
			SpinGuard  lock(region->lock);
			const auto page_begin = Common::AlignDown(chunk_begin, PAGE_SIZE);
			const auto page_end   = Common::AlignUp(chunk_end, PAGE_SIZE);
			Protect(page_begin, page_end - page_begin, Common::VirtualMemory::Mode::ReadWrite);
			std::memcpy(destination, source, chunk_end - chunk_begin);
			const auto last = static_cast<size_t>((page_end - region_base) / PAGE_SIZE);
			for (auto first = static_cast<size_t>((page_begin - region_base) / PAGE_SIZE); first < last;) {
				const auto perms = region->pages[first].Perms();
				auto       next  = first + 1;
				while (next < last && region->pages[next].Perms() == perms) {
					++next;
				}
				if (perms != Common::VirtualMemory::Mode::ReadWrite) {
					Protect(region_base + first * PAGE_SIZE, (next - first) * PAGE_SIZE, perms);
				}
				first = next;
			}
			chunk_begin = chunk_end;
		}
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	Common::SrwMutex                        region_mutex;
	// Regions with pending pages (see Region::pending).
	std::mutex                                deferred_mutex;
	std::vector<std::pair<Region*, uint64_t>> deferred;
	std::atomic<bool>                         has_deferred {false};
};

static_assert(std::atomic<void*>::is_always_lock_free);

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {}

PageManager::~PageManager() = default;

template <bool track, bool is_read>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	m_impl->UpdatePageWatchers<track, is_read>(vaddr, size);
}

void PageManager::WriteGuest(uint64_t vaddr, const void* data, uint64_t size) {
	m_impl->WriteGuest(vaddr, data, size);
}

template void PageManager::UpdatePageWatchers<true, true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<true, false>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false, true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false, false>(uint64_t, uint64_t);

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask, bool defer) {
	const auto start_range = mask.FirstRange();
	const auto end_range   = mask.LastRange();
	const auto first       = start_range.first;
	const auto last        = end_range.second;
	if (start_range.second == end_range.second) {
		m_impl->UpdatePageWatchers<track, is_read>(base_addr + first * PAGE_SIZE,
		                                           (last - first) * PAGE_SIZE, defer);
		return;
	}

	auto* region = track ? m_impl->GetOrCreateRegion(base_addr) : m_impl->FindRegion(base_addr);
	if (region == nullptr) {
		Fatal("untracking unknown region 0x%016" PRIx64, base_addr);
	}
	m_impl->UpdateRegionWatchers<track, is_read, true>(*region, base_addr, first, last, &mask,
	                                                   defer);
}

void PageManager::FlushDeferred() {
	m_impl->FlushDeferred();
}

bool PageManager::HasDeferred() const noexcept {
	return m_impl->HasDeferred();
}

template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&, bool);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&, bool);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&, bool);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&, bool);

} // namespace Libs::Graphics
