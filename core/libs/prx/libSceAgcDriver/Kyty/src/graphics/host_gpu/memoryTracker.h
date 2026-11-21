#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/fastTls.h"
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
#include <type_traits>
#include <vector>

namespace Libs::Graphics {

class MemoryTracker final {
public:
	explicit MemoryTracker(PageManager& page_manager);

	[[nodiscard]] PageManager& Pages() noexcept { return m_page_manager; }
	~MemoryTracker();

	KYTY_CLASS_NO_COPY(MemoryTracker);

	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// A page of the range is rewritten so often that it is left unwatched: no fault reports its writes.
	[[nodiscard]] bool IsRegionUnwatched(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
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

	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRange(uint64_t vaddr, uint64_t size, bool is_written, RangeFunc&& range_func,
	                        UploadFunc&& upload_func) {
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
		if (memo.tracker == m_id && memo.vaddr == vaddr && memo.size == size &&
		    memo.written == is_written && (memo.epoch == 0 || memo.epoch == epoch) &&
		    memo.generations == RegionGenerations(vaddr, size)) {
			upload_func();
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
			manager->ForEachModifiedRange<DirtySource::Cpu, true>(offset, bytes, range_func);
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
	}

private:
	static constexpr size_t REGION_COUNT = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	static_assert(REGION_COUNT % 64 == 0);

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

	// Sum of the generations of the regions the range touches, without their locks; 0 when one
	// of them does not exist yet. Indexed like Iterate (AnyPS5: by TrackerOffset).
	[[nodiscard]] uint64_t RegionGenerations(uint64_t vaddr, uint64_t size) const noexcept {
		uint64_t   sum   = 0;
		const auto first = TrackerOffset(vaddr) / TRACKER_REGION_SIZE;
		const auto last  = TrackerOffset(vaddr + size - 1) / TRACKER_REGION_SIZE;
		for (auto index = first; index <= last; index++) {
			const auto* manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr) {
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

	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	Common::SrwMutex                               m_region_mutex;
	PageManager&                                   m_page_manager;
	// Tells upload memos of a destroyed tracker from this one's.
	const uint64_t m_id = s_next_id.fetch_add(1, std::memory_order_relaxed);
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
