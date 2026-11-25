#include "graphics/host_gpu/memoryTracker.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"

namespace Libs::Graphics {

static_assert(std::atomic<void*>::is_always_lock_free);

MemoryTracker::MemoryTracker(PageManager& page_manager): m_page_manager(page_manager) {
	m_regions = std::make_unique<std::atomic<RegionManager*>[]>(REGION_COUNT);
	m_gpu_regions = std::make_unique<std::atomic<uint64_t>[]>(REGION_COUNT / 64);
}

MemoryTracker::~MemoryTracker() = default;

#if KYTY_BUILD == KYTY_BUILD_DEBUG
void MemoryTracker::ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                          const char* operation) const noexcept {
	if (!GuestRange {vaddr, size}.Valid() || (vaddr & (TRACKER_PAGE_SIZE - 1)) != 0 ||
	    (size & (TRACKER_PAGE_SIZE - 1)) != 0) {
		EXIT("MemoryTracker: invalid dirty-page validation range\n");
	}
	for (auto page = vaddr; page < vaddr + size; page += TRACKER_PAGE_SIZE) {
		if (!dirty.Intersects(page, TRACKER_PAGE_SIZE)) {
			EXIT("MemoryTracker: GPU-dirty tracker page has no dirty bytes, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}

void MemoryTracker::ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                              const char* operation) {
	ValidateRange(vaddr, size);
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto page = begin; page < end; page += TRACKER_PAGE_SIZE) {
		const bool has_dirty_bytes = dirty.Intersects(page, TRACKER_PAGE_SIZE);
		if (IsRegionGpuModified(page, TRACKER_PAGE_SIZE) != has_dirty_bytes) {
			EXIT("MemoryTracker: tracker and byte ownership disagree, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}
#endif

void MemoryTracker::ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("invalid memory tracker range\n");
	}
}

RegionManager* MemoryTracker::GetOrCreateRegion(uint64_t index) {
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	std::lock_guard lock(m_region_mutex);
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	auto  manager = std::make_unique<RegionManager>(m_page_manager, TrackerAddress(index * TRACKER_REGION_SIZE), &m_hot);
	auto* ptr     = manager.get();
	m_region_storage.push_back(std::move(manager));
	m_regions[index].store(ptr, std::memory_order_release);
	return ptr;
}

// Queries take no lock: dirty words are read atomically, and a query runs on the GPU thread, which
// alone writes GPU-dirty bits while other threads only set CPU-dirty ones, so it sees one instant.
bool MemoryTracker::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	if (Common::DebugCounters::g_aps5_stats) g_cpumod_calls.fetch_add(1, std::memory_order_relaxed);
	if (Common::DebugCounters::g_aps5_stats) g_cpumod_regions.fetch_add((vaddr % TRACKER_REGION_SIZE + size + TRACKER_REGION_SIZE - 1) / TRACKER_REGION_SIZE, std::memory_order_relaxed);
	return Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		return manager->IsModified<DirtySource::Cpu>(offset, bytes);
	});
}

bool MemoryTracker::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	if (Common::DebugCounters::g_aps5_stats) g_gpumod_calls.fetch_add(1, std::memory_order_relaxed);
	if (size == 0 || !MayHoldGpuWrites(vaddr, size)) {
		return false;
	}
	if (Common::DebugCounters::g_aps5_stats) g_gpumod_regions.fetch_add((vaddr % TRACKER_REGION_SIZE + size + TRACKER_REGION_SIZE - 1) / TRACKER_REGION_SIZE, std::memory_order_relaxed);
	return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		return manager->IsModified<DirtySource::Gpu>(offset, bytes);
	});
}

bool MemoryTracker::IsRegionUnwatched(uint64_t vaddr, uint64_t size) {
	return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		return manager->IsHot(offset, bytes);
	});
}

void MemoryTracker::MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size) {
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Cpu, true>(offset, bytes);
	});
}

void MemoryTracker::UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		// Nothing to clear in a GPU-clean range; checked without the lock like the queries.
		if (!manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
			return;
		}
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, false>(offset, bytes);
	});
}

void MemoryTracker::UnmarkRegionsAsGpuModified(std::span<const GuestRange> ranges) {
	auto& chunks = ::ShaderRecompiler::ThreadOwned(s_unmark_chunks);
	chunks.clear();
	for (size_t index = 0; index < ranges.size(); index++) {
		Iterate<false>(ranges[index].address, ranges[index].size,
		               [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			               // Nothing to clear in a GPU-clean range; checked without the lock like
			               // the queries.
			               if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
				               chunks.push_back({manager, index, offset, bytes});
			               }
		               });
	}
	for (size_t first = 0; first < chunks.size();) {
		auto* manager = chunks[first].manager;
		auto  last    = first + 1;
		while (last < chunks.size() && chunks[last].manager == manager) {
			last++;
		}
		std::scoped_lock lock(manager->lock);
		manager->ClearGpuModifiedBatch(last - first, [&](size_t chunk) {
			return std::pair<uint64_t, uint64_t> {chunks[first + chunk].offset,
			                                      chunks[first + chunk].bytes};
		});
		first = last;
	}
}

} // namespace Libs::Graphics
