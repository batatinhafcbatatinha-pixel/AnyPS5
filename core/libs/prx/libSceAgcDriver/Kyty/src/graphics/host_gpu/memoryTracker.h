#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/fastTls.h"
#include "../shader/recompiler/ThreadOwned.hpp"
#include "common/assert.h"
#include "common/srwLock.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class MemoryTracker final {
public:
	explicit MemoryTracker(PageManager& page_manager);

	[[nodiscard]] PageManager& Pages() noexcept { return m_page_manager; }
	// Exact hot pages of this tracker (see RegionManager and HotPages).
	[[nodiscard]] HotPages& Hot() noexcept { return m_hot; }
	// GPU thread: the shadow slot of the exact hot page holding `address`, HotPages::NO_SLOT
	// when it is watched normally (see RegionManager::ExactHotSlot).
	[[nodiscard]] uint16_t HotSlot(uint64_t address) const noexcept {
		const auto* manager =
		    m_regions[TrackerOffset(address) / TRACKER_REGION_SIZE].load(std::memory_order_acquire);
		return manager != nullptr ? manager->ExactHotSlot(address % TRACKER_REGION_SIZE)
		                          : HotPages::NO_SLOT;
	}

	// GPU thread, before each upload of bytes shaders read through device addresses (which no
	// query reaches): func(page) for each exact hot page whose bytes differ from its upload, so
	// the upload takes it as a fault would have queued it. Once per frame, hot pages unchanged
	// for HOT_IDLE_FRAMES frames, or all of them when the compares of the last frame cost more
	// than they saved (HotPages::EndFrame), are demoted; func(page) gets them too, as their next
	// upload watches them again.
	template <typename Func>
	void ScanHotPages(Func&& func) {
		if (!ExactHotPagesEnabled()) {
			return;
		}
		m_hot.Recycle();
		const auto frame    = TrackerFrame();
		const bool new_frame = m_hot_scan_frame != frame;
		const bool over      = m_hot.EndFrame(frame);
		m_hot_scan_frame     = frame;
		if (m_hot.Live() == 0) {
			return;
		}
		m_hot.ForEachLive([&](uint16_t slot, uint64_t page) {
			auto* manager = m_regions[TrackerOffset(page) / TRACKER_REGION_SIZE].load(std::memory_order_acquire);
			if (manager == nullptr) {
				return;
			}
			const auto offset = page % TRACKER_REGION_SIZE;
			if (new_frame && (over || frame - m_hot.ChangedFrame(slot) > HOT_IDLE_FRAMES)) {
				std::scoped_lock lock(manager->lock);
				if (manager->DemoteExactHotBytes(offset, TRACKER_PAGE_SIZE)) {
					func(page);
				}
				return;
			}
			if (manager->ExactHotChangedAt(offset)) {
				func(page);
			}
		});
	}
	~MemoryTracker();

	KYTY_CLASS_NO_COPY(MemoryTracker);

	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// A page of the range is rewritten so often that it is left unwatched: no fault reports its writes.
	[[nodiscard]] bool IsRegionUnwatched(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	// UnmarkRegionAsGpuModified for each of `ranges` (ascending, disjoint), the ranges of one
	// tracker region cleared under one lock with one protection change (see
	// RegionManager::ClearGpuModifiedBatch).
	void UnmarkRegionsAsGpuModified(std::span<const GuestRange> ranges);
	// Removes protection from a range and flushes GPU-owned data when required.
	template <typename Flush>
	void InvalidateRegion(uint64_t vaddr, uint64_t size, Flush&& on_flush) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				// Perform both the GPU modification check and CPU state change with the lock in
				// case the GPU thread is racing to mark the page modified. If a flush is needed,
				// on_flush performs the CPU state change.
				std::scoped_lock lock(manager->lock);
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				manager->ChangeState<DirtySource::Cpu, true>(offset, bytes);
				return false;
			}();
			if (should_flush) {
				on_flush();
			}
		});
	}
#if KYTY_BUILD == KYTY_BUILD_DEBUG
	void ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                           const char* operation) const noexcept;
	void ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                               const char* operation);
#else
	void ValidateGpuDirtyPages(const RangeSet&, uint64_t, uint64_t, const char*) const noexcept {}
	void ValidateGpuDirtyOwnership(const RangeSet&, uint64_t, uint64_t, const char*) {}
#endif

	template <bool clear, typename Func>
	void ForEachDownloadRange(uint64_t vaddr, uint64_t size, Func&& func) {
		static_assert(std::is_nothrow_invocable_v<Func&, uint64_t, uint64_t>);
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			// Lock-free like the queries (see IsRegionGpuModified): a range without GPU-dirty pages
			// has nothing to download, and need not wait for a write fault holding the region.
			if (!manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
				return;
			}
			std::scoped_lock lock(manager->lock);
			manager->ForEachModifiedRange<DirtySource::Gpu, false>(offset, bytes, func);
			if constexpr (clear) {
				manager->ChangeState<DirtySource::Gpu, false>(offset, bytes);
			}
		});
	}

	// generations_out, when given, gets the sum of the range's region generations as the call
	// leaves them (see RegionGenerations), read race-free: an equal sum later means the call
	// would find nothing to upload and change nothing again.
	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRange(uint64_t vaddr, uint64_t size, bool is_written, RangeFunc&& range_func,
	                        UploadFunc&& upload_func, uint64_t* generations_out = nullptr) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, uint64_t, uint64_t>);
		static_assert(std::is_nothrow_invocable_v<UploadFunc&>);
		// Draws bind the same ranges again and again, unchanged. A call leaves its range
		// CPU-clean (except hot pages already uploaded in this submission) and, when written,
		// GPU-dirty with its regions noted in m_gpu_regions (never cleared). The same call again,
		// with no change to the range's regions since (their generations sum to the same value),
		// finds nothing to upload and changes nothing, so it only runs the upload callback.
		// Regions with hot pages are clean only for the submission they were uploaded in, so
		// their memo also holds that submission's epoch.
		const auto epoch = StickyPagesEnabled() ? TrackerEpoch() : UPLOAD_MEMO_NO_STICKY;
		auto&      memo  = UploadMemoSlot(vaddr, size, is_written);
		if (ExactHotPagesEnabled()) {
			// No upload of an earlier call still copies into a shadow (see HotPages).
			m_hot.Recycle();
		}
		if (memo.tracker == m_id && memo.vaddr == vaddr && memo.size == size &&
		    memo.written == is_written && (memo.epoch == 0 || memo.epoch == epoch) &&
		    memo.generations == RegionGenerations(vaddr, size)) {
			upload_func();
			if (generations_out != nullptr) {
				*generations_out = memo.generations;
			}
			return;
		}
		// The regions' generations as this call leaves them. A region skipped without its lock is
		// read before its state, so a change made meanwhile shows up as a newer generation.
		uint64_t generations = 0;
		bool     hot         = false;
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			if (!is_written) {
				const auto generation = manager->Generation();
				if (!manager->IsModified<DirtySource::Cpu>(offset, bytes)) {
					generations += generation;
					hot |= manager->HasHot();
					return;
				}
			}
			manager->lock.lock();
			if (is_written) {
				// The range becomes GPU-modified below: its exact hot pages are uploaded as
				// ordinary dirty pages, watched before their bytes are copied, and stay watched.
				(void)manager->DemoteExactHotBytes(offset, bytes);
			}
			manager->ForEachModifiedRange<DirtySource::Cpu, true>(offset, bytes, range_func,
			                                                      !is_written);
			if (!is_written) {
				generations += manager->Generation();
				hot |= manager->HasHot();
				manager->lock.unlock();
			}
		});
		upload_func();
		if (is_written) {
			NoteGpuWrittenRegions(vaddr, size);
			Iterate<false>(vaddr, size,
			               [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				               manager->ChangeState<DirtySource::Gpu, true>(offset, bytes);
				               generations += manager->Generation();
				               hot |= manager->HasHot();
				               manager->lock.unlock();
			               });
		}
		memo = {.tracker     = m_id,
		        .vaddr       = vaddr,
		        .size        = size,
		        .generations = generations,
		        .epoch       = hot ? epoch : 0,
		        .written     = is_written};
		if (generations_out != nullptr) {
			*generations_out = generations;
		}
	}

	// ForEachUploadRange(range, false, ...) for each of `ranges` with an empty upload callback,
	// func(index, address, bytes) getting the runs of ranges[index] in the same order. The ranges
	// of one tracker region are re-watched with one protection change: the separate calls make
	// one protection call per run at least, the batch one per run of pages it leaves watched. As
	// in the separate calls, every page is watched again before func sees it, so a guest write
	// after that faults and queues the page again, and one before it is in the bytes copied later.
	template <typename Func>
	// defer: see RegionManager::ForEachCpuModifiedRangeBatch (not with the rewritten-page mode,
	// whose uploads watch their pages at once).
	void ForEachUploadRangeBatch(std::span<const GuestRange> ranges, Func&& func, bool defer = false) {
		static_assert(std::is_nothrow_invocable_v<Func&, size_t, uint64_t, uint64_t>);
		if (StickyPagesEnabled()) {
			for (size_t index = 0; index < ranges.size(); index++) {
				ForEachUploadRange(
				    ranges[index].address, ranges[index].size, false,
				    [&](uint64_t address, uint64_t bytes) noexcept { func(index, address, bytes); },
				    []() noexcept {});
			}
			return;
		}
		if (ExactHotPagesEnabled()) {
			// No upload of an earlier call still copies into a shadow (see HotPages).
			m_hot.Recycle();
		}
		auto& chunks = ::ShaderRecompiler::ThreadOwned(s_batch_chunks);
		chunks.clear();
		for (size_t index = 0; index < ranges.size(); index++) {
			Iterate<true>(ranges[index].address, ranges[index].size,
			              [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				              chunks.push_back({manager, index, offset, bytes});
			              });
		}
		for (size_t first = 0; first < chunks.size();) {
			auto* manager = chunks[first].manager;
			auto  last    = first + 1;
			while (last < chunks.size() && chunks[last].manager == manager) {
				last++;
			}
			// Lock-free like ForEachUploadRange: a region with nothing CPU-dirty in the ranges
			// changes nothing.
			bool modified = false;
			for (auto chunk = first; chunk < last && !modified; chunk++) {
				modified = manager->IsModified<DirtySource::Cpu>(chunks[chunk].offset,
				                                                 chunks[chunk].bytes);
			}
			if (modified) {
				std::scoped_lock lock(manager->lock);
				manager->ForEachCpuModifiedRangeBatch(
				    last - first,
				    [&](size_t chunk) {
					    return std::pair<uint64_t, uint64_t> {chunks[first + chunk].offset,
					                                          chunks[first + chunk].bytes};
				    },
				    [&](size_t chunk, uint64_t address, uint64_t bytes) {
					    func(chunks[first + chunk].range, address, bytes);
				    },
				    defer);
			}
			first = last;
		}
	}

private:
	static constexpr size_t REGION_COUNT = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	static_assert(REGION_COUNT % 64 == 0);

	// ForEachUploadRangeBatch's ranges cut at tracker regions, per thread.
	struct BatchChunk {
		RegionManager* manager;
		size_t         range;
		uint64_t       offset;
		uint64_t       bytes;
	};
	// AnyPS5: no thread_local with a destructor in this module (emulated TLS on MinGW).
	inline static thread_local std::vector<BatchChunk>* s_batch_chunks = nullptr;
	// UnmarkRegionsAsGpuModified's ranges cut at tracker regions, per thread.
	inline static thread_local std::vector<BatchChunk>* s_unmark_chunks = nullptr;

	// ForEachUploadRange's last call per range, per thread.
	struct UploadMemo {
		uint64_t tracker     = 0;
		uint64_t vaddr       = 0;
		uint64_t size        = 0;
		uint64_t generations = 0;
		uint64_t epoch       = 0; // 0: valid in every submission
		bool     written     = false;
	};
	static constexpr size_t   UPLOAD_MEMO_SLOTS     = 4096;
	static constexpr uint64_t UPLOAD_MEMO_NO_STICKY = ~uint64_t {0};
	// Per thread, in a Windows TLS slot: MinGW's thread_local is emulated (pthread_getspecific on
	// every access), which showed up in the GPU thread's per-draw buffer lookups.
	inline static Common::FastTls<std::array<UploadMemo, UPLOAD_MEMO_SLOTS>*> s_upload_memo;
	inline static std::atomic<uint64_t> s_next_id {1};

	[[nodiscard]] static UploadMemo& UploadMemoSlot(uint64_t vaddr, uint64_t size, bool written) {
		auto* memo = s_upload_memo.Get();
		if (memo == nullptr) {
			memo = new std::array<UploadMemo, UPLOAD_MEMO_SLOTS>();
			s_upload_memo.Set(memo);
		}
		auto hash = (vaddr >> 4u) ^ (size * 0x9e3779b97f4a7c15ull) ^ (written ? 1u : 0u);
		hash ^= hash >> 29u;
		return (*memo)[hash % UPLOAD_MEMO_SLOTS];
	}

public:
	// Sum of the generations of the regions the range touches, without their locks; 0 when one
	// of them does not exist yet, or when the range holds an exact hot page: guest writes change
	// such a page without a fault, so no generation tells that it stayed the same. Indexed like
	// Iterate (AnyPS5: by TrackerOffset).
	[[nodiscard]] uint64_t RegionGenerations(uint64_t vaddr, uint64_t size) const noexcept {
		uint64_t   sum   = 0;
		const auto first = TrackerOffset(vaddr) / TRACKER_REGION_SIZE;
		const auto last  = TrackerOffset(vaddr + size - 1) / TRACKER_REGION_SIZE;
		for (auto index = first; index <= last; index++) {
			const auto* manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr) {
				return 0;
			}
			const auto region_begin = index * TRACKER_REGION_SIZE;
			const auto offset       = TrackerOffset(vaddr);
			const auto begin        = std::max(offset, region_begin);
			const auto end          = std::min(offset + size, region_begin + TRACKER_REGION_SIZE);
			if (manager->HasExactHotBytes(begin - region_begin, end - begin)) {
				return 0;
			}
			sum += manager->Generation();
		}
		return sum;
	}

	// AnyPS5: one bit per region that ever held GPU-written pages (set before the pages' bits).
	// Most GPU-modified checks (several per draw) end on this bit without touching the region.
	void NoteGpuWrittenRegions(uint64_t vaddr, uint64_t size) noexcept {
		const auto first = TrackerOffset(vaddr) / TRACKER_REGION_SIZE;
		const auto last  = TrackerOffset(vaddr + size - 1) / TRACKER_REGION_SIZE;
		for (auto index = first; index <= last; ++index) {
			const auto bit = uint64_t {1} << (index % 64);
			if ((m_gpu_regions[index / 64].load(std::memory_order_relaxed) & bit) == 0) {
				m_gpu_regions[index / 64].fetch_or(bit, std::memory_order_seq_cst);
			}
		}
	}
	[[nodiscard]] bool MayHoldGpuWrites(uint64_t vaddr, uint64_t size) const noexcept {
		const auto first = TrackerOffset(vaddr) / TRACKER_REGION_SIZE;
		const auto last  = TrackerOffset(vaddr + size - 1) / TRACKER_REGION_SIZE;
		for (auto index = first; index <= last; ++index) {
			if ((m_gpu_regions[index / 64].load(std::memory_order_seq_cst) >> (index % 64)) & 1u) {
				return true;
			}
		}
		return false;
	}
	std::unique_ptr<std::atomic<uint64_t>[]> m_gpu_regions;

private:
	template <bool create, typename Func>
	bool Iterate(uint64_t vaddr, uint64_t size, Func&& func) {
		ValidateRange(vaddr, size);
		using Result = std::invoke_result_t<Func, RegionManager*, uint64_t, uint64_t>;
		constexpr bool returns_bool = std::is_same_v<Result, bool>;
		uint64_t       remaining    = size;
		uint64_t       index        = TrackerOffset(vaddr) / TRACKER_REGION_SIZE;
		uint64_t       offset       = vaddr % TRACKER_REGION_SIZE;
		while (remaining != 0) {
			const auto bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr && create) {
				manager = GetOrCreateRegion(index);
			}
			if (manager != nullptr) {
				if constexpr (returns_bool) {
					if (func(manager, offset, bytes)) {
						return true;
					}
				} else {
					func(manager, offset, bytes);
				}
			}
			remaining -= bytes;
			offset = 0;
			index++;
		}
		return false;
	}

	static void    ValidateRange(uint64_t vaddr, uint64_t size);
	RegionManager* GetOrCreateRegion(uint64_t index);

	// Frames a hot page may go unchanged before it is watched again (ScanHotPages).
	static constexpr uint64_t HOT_IDLE_FRAMES = 8;

	HotPages                                       m_hot;
	uint64_t                                       m_hot_scan_frame = 0; // GPU thread
	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	Common::SrwMutex                               m_region_mutex;
	PageManager&                                   m_page_manager;
	// Tells upload memos of a destroyed tracker from this one's.
	const uint64_t m_id = s_next_id.fetch_add(1, std::memory_order_relaxed);
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
