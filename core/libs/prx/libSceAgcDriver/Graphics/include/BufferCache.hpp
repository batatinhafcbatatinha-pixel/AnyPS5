#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BUFFERCACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BUFFERCACHE_HPP

// Device-local cache of guest buffer memory (APS5_BUFFER_CACHE=1). The design follows KytyPS5's
// buffer cache (src/graphics/host_gpu/renderer/cache/bufferCache.cpp at commit 87de64e0, GPL-2.0):
// guest ranges live in device-local buffers, only the parts the CPU (or other work) changed since
// they were last made current are uploaded, and the GPU reads video memory instead of host memory.
// Unlike KytyPS5, change tracking is AnyPS5's own write tracker (64 KiB blocks stamped by write-watch
// collects and by the driver's records of GPU writes) and an upload is a GPU copy from the range's
// host import, recorded in queue order, so no CPU copy is made.

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace AgcDriver::Graphics {

class Buffer;
class Recorder;

bool BufferCacheEnabled();

struct CachedBufferRange {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
};

// Under the GPU mutex, with `recorder` the active one. Returns a device-local binding of
// [address, address + bytes) whose bytes, by the time later recorded work runs, are the newest guest
// bytes (pending image results and resident copies over the range are flushed into guest memory
// first, then the changed blocks are copied in). Nothing when the range cannot be cached (no host
// import, outside the write-watched arena, crossing a cache chunk, over the budget, written by queued
// labels or copied writers) or when the upload would end an open render pass for data that changes
// too often to be worth it: the caller then uses its old path. `behindGpuWrite` is set when recorded
// GPU work writes the range (the caller's read then needs its barrier after the upload, which this
// call records).
std::optional<CachedBufferRange> ObtainCachedBuffer(const Context& context, Recorder& recorder, std::uint64_t address, std::size_t bytes, bool* behindGpuWrite = nullptr);

// Guest -> device address table for address-based (GPU-side SRT) shaders, KytyPS5's BDA page table
// made two-level: the root (BufferCacheTable) holds BufferCacheTableRoots device addresses of leaves,
// one per 2^BufferCacheTableLeafBits bytes of guest address space (0: no leaf); a leaf holds one
// device address per 2^BufferCacheTablePageBits-byte guest page (0: not resident). A shader finds
// guest address A at
//   leaf = root[A >> LeafBits]; page = leaf[(A & (2^LeafBits - 1)) >> PageBits];
//   device address = page + (A & (2^PageBits - 1))   (fall back to the old path when 0)
// Pages are mapped per 4 MiB cache chunk. The table and the chunks are read-only for shaders: GPU
// writes through them would not reach guest memory (not implemented; use the old path for stores).
constexpr std::uint32_t BufferCacheTablePageBits = 14;
constexpr std::uint32_t BufferCacheTableLeafBits = 30;
constexpr std::uint64_t BufferCacheTableRoots = 1024;
// The root's device address (0 without buffer device addresses or with the cache off). Under the GPU
// mutex; records the root's clear the first time.
VkDeviceAddress BufferCacheTable(const Context& context, Recorder& recorder);
// Makes [address, address + bytes) resident for later recorded work: its chunks exist, their table
// entries are written, and their blocks changed since the last upload are copied in (ending an open
// render pass if needed). False when a part could not be (no host import, untracked, over budget):
// those pages keep a 0 entry. Under the GPU mutex.
bool MakeResident(const Context& context, Recorder& recorder, std::uint64_t address, std::size_t bytes);
// Pages address-based shaders read through the old range lookup (BdaAbi accessed-page records, read
// after their batch): queued from any thread, made resident by the next MakeQueuedResident (under the
// GPU mutex), so later work finds them in the table.
void QueueResident(std::uint64_t address, std::size_t bytes);
void MakeQueuedResident(const Context& context, Recorder& recorder);
// Uploads the blocks of every resident range that changed since their last upload (KytyPS5's
// SynchronizeBda before work that reads through the table): call under the GPU mutex before recording
// work that reads through the table, so it never reads bytes older than guest memory at that point.
void SyncResident(const Context& context, Recorder& recorder);

// Forgets every cached chunk of the device (device teardown).
void ClearBufferCache(VkDevice device);

}

#endif
