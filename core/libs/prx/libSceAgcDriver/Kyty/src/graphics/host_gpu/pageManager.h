#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <memory>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute };

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	template <bool track, bool is_read = false>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	// defer: the watcher counts change now, but the OS protection of the pages whose protection
	// changes is applied by the next FlushDeferred, not here. Only for changes that make pages
	// stricter, where the weaker protection meanwhile is harmless to the caller (see
	// BufferCache::FlushDeferredTracking).
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask, bool defer = false);

	// Applies the protection the watcher counts give to every page a deferred change left
	// behind, in as few calls as the pages' protections allow.
	void               FlushDeferred();
	[[nodiscard]] bool HasDeferred() const noexcept;

	// Stores bytes at their guest addresses without raising guest faults: the pages are writable
	// for the copy and get their watcher protection back. For GPU results in ranges the backing
	// store does not hold (private runtime mappings); the backing store takes the others.
	void WriteGuest(uint64_t vaddr, const void* data, uint64_t size);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
