// Device-local guest buffer cache. Adapted in design from KytyPS5
// src/graphics/host_gpu/renderer/cache/bufferCache.cpp (ObtainBuffer / SynchronizeBuffer /
// UploadCopies / Register's BDA page-table entries) at commit 87de64e0 (GPL-2.0, see
// THIRD_PARTY_NOTICES.md): device-local buffers over guest ranges, uploads of the changed parts only
// behind one barrier pair, and a page table of their device addresses for address-based shaders. The
// change tracking, chunking, upload source (the range's host import, copied on the GPU), the
// two-level table and the render-pass policy are AnyPS5's.
#include "prx/libSceAgcDriver/Graphics/include/BufferCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace AgcDriver::Graphics {

namespace {

constexpr std::uint64_t ChunkBytes = std::uint64_t{4} << 20u;
constexpr std::uint64_t BlockBytes = std::uint64_t{64} << 10u;
constexpr std::size_t BlocksPerChunk = static_cast<std::size_t>(ChunkBytes / BlockBytes);
constexpr std::uint64_t TablePageBytes = std::uint64_t{1} << BufferCacheTablePageBits;
constexpr std::uint64_t LeafSpan = std::uint64_t{1} << BufferCacheTableLeafBits;
constexpr std::uint64_t LeafEntries = LeafSpan / TablePageBytes;

// Ranges crossing a chunk boundary (whole vertex buffers) get a span buffer of their own, up to this
// size; spans are only bound, never put into the address table.
constexpr std::uint64_t SpanMaxBytes = std::uint64_t{64} << 20u;

struct Chunk {
    std::shared_ptr<Buffer> buffer;
    std::uint64_t bytes = 0;
    // The tracker generation each block was last uploaded at (0: never, or only partly).
    std::vector<std::uint64_t> uploaded;
    // Uses and uploads per block, saturating: decides whether an upload may end an open pass.
    std::vector<std::uint16_t> uses;
    std::vector<std::uint16_t> uploads;
    std::array<bool, ChunkBytes / TablePageBytes> mapped{};
};

struct Stats {
    std::uint64_t obtained = 0;
    std::uint64_t clean = 0;
    std::uint64_t uploaded = 0;
    std::uint64_t uploadBytes = 0;
    std::uint64_t passBreaks = 0;
    std::uint64_t refusedInPass = 0;
    std::uint64_t crossing = 0;
    std::uint64_t noImport = 0;
    std::uint64_t untracked = 0;
    std::uint64_t overBudget = 0;
    std::uint64_t resident = 0;
    std::uint64_t residentRefused = 0;
    std::uint64_t residentSourced = 0;
    std::uint64_t refusedRewritten = 0;
};

struct Cache {
    std::mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    std::unordered_map<std::uint64_t, Chunk> chunks;
    // Span buffers by base (they may overlap: each is a read-only copy of guest memory).
    std::map<std::uint64_t, Chunk> spans;
    std::uint64_t spanBytes = 0;
    std::shared_ptr<Buffer> root;
    std::unordered_map<std::uint64_t, std::shared_ptr<Buffer>> leaves;
    // Resident ranges (begin -> end, merged), re-synced by SyncResident.
    std::map<std::uint64_t, std::uint64_t> residentRanges;
    Stats stats;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

Cache& cache() {
    static auto* instance = new Cache();
    return *instance;
}

std::size_t budgetChunks() {
    static const std::size_t chunks = [] {
        const char* text = std::getenv("APS5_BUFFER_CACHE_MIB");
        const auto mib = text != nullptr ? std::strtoull(text, nullptr, 10) : 1024ull;
        return static_cast<std::size_t>(std::max<unsigned long long>(mib, 4ull) / (ChunkBytes >> 20u));
    }();
    return chunks;
}

std::uint64_t cachedBytes(const Cache& state);

void report(Cache& state) {
    const auto now = std::chrono::steady_clock::now();
    if (now - state.lastReport < std::chrono::seconds(10)) return;
    state.lastReport = now;
    const auto& s = state.stats;
    std::fprintf(stderr, "[bufcache] (10 s) %llu obtained: %llu clean, %llu uploaded (%.1f MiB, %llu ended a pass), refused: %llu in an open pass, %llu crossing a chunk, %llu without import, %llu untracked, %llu over budget; resident requests %llu (%llu refused); %llu read from resident staging, %llu refused as rewritten; %zu chunks (%llu MiB), %zu table leaves\n", static_cast<unsigned long long>(s.obtained), static_cast<unsigned long long>(s.clean), static_cast<unsigned long long>(s.uploaded), static_cast<double>(s.uploadBytes) / (1024.0 * 1024.0), static_cast<unsigned long long>(s.passBreaks), static_cast<unsigned long long>(s.refusedInPass), static_cast<unsigned long long>(s.crossing), static_cast<unsigned long long>(s.noImport), static_cast<unsigned long long>(s.untracked), static_cast<unsigned long long>(s.overBudget), static_cast<unsigned long long>(s.resident), static_cast<unsigned long long>(s.residentRefused), static_cast<unsigned long long>(s.residentSourced), static_cast<unsigned long long>(s.refusedRewritten), state.chunks.size() + state.spans.size(), static_cast<unsigned long long>(cachedBytes(state) >> 20u), state.leaves.size());
    state.stats = {};
}

// Data the title rewrites about every use (rings) is not worth ending a render pass for: its draw
// keeps the old path. A block uploaded at most once, or at most once per four uses, is.
bool worthPassBreak(const Chunk& chunk, std::size_t block) {
    return chunk.uploads[block] < 2 || static_cast<std::uint32_t>(chunk.uploads[block]) * 4u <= chunk.uses[block];
}

void resetFor(Cache& state, VkDevice device) {
    if (state.device == device) return;
    state.chunks.clear();
    state.spans.clear();
    state.spanBytes = 0;
    state.leaves.clear();
    state.root.reset();
    state.residentRanges.clear();
    state.device = device;
}

Chunk makeChunk(const Context& context, std::uint64_t bytes) {
    Chunk chunk;
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (context.bufferDeviceAddress) usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    chunk.buffer = std::make_shared<Buffer>(context, static_cast<std::size_t>(bytes), usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    chunk.bytes = bytes;
    const auto blocks = static_cast<std::size_t>(bytes / BlockBytes);
    chunk.uploaded.assign(blocks, 0);
    chunk.uses.assign(blocks, 0);
    chunk.uploads.assign(blocks, 0);
    return chunk;
}

std::uint64_t cachedBytes(const Cache& state) {
    return state.chunks.size() * ChunkBytes + state.spanBytes;
}

Chunk* chunkFor(Cache& state, const Context& context, std::uint64_t chunkBase) {
    if (auto found = state.chunks.find(chunkBase); found != state.chunks.end()) return &found->second;
    if (cachedBytes(state) + ChunkBytes > budgetChunks() * ChunkBytes) {
        ++state.stats.overBudget;
        return nullptr;
    }
    return &state.chunks.emplace(chunkBase, makeChunk(context, ChunkBytes)).first->second;
}

// The span holding [begin, end), made (chunk-aligned) when none does; spans it covers wholly are
// dropped (their buffers kept by the recorder for work already recorded).
Chunk* spanFor(Cache& state, const Context& context, Recorder& recorder, std::uint64_t begin, std::uint64_t end, std::uint64_t& spanBase) {
    for (auto it = state.spans.upper_bound(begin); it != state.spans.begin();) {
        --it;
        if (it->first + SpanMaxBytes <= begin) break;
        if (it->first <= begin && end <= it->first + it->second.bytes) {
            spanBase = it->first;
            return &it->second;
        }
    }
    spanBase = begin / ChunkBytes * ChunkBytes;
    const auto spanEnd = (end + ChunkBytes - 1) / ChunkBytes * ChunkBytes;
    const auto bytes = spanEnd - spanBase;
    if (bytes > SpanMaxBytes) {
        ++state.stats.crossing;
        return nullptr;
    }
    for (auto it = state.spans.lower_bound(spanBase); it != state.spans.end() && it->first < spanEnd;) {
        if (it->first + it->second.bytes > spanEnd) {
            ++it;
            continue;
        }
        recorder.Keep(it->second.buffer, 0);
        state.spanBytes -= it->second.bytes;
        it = state.spans.erase(it);
    }
    if (cachedBytes(state) + bytes > budgetChunks() * ChunkBytes) {
        ++state.stats.overBudget;
        return nullptr;
    }
    state.spanBytes += bytes;
    return &state.spans.insert_or_assign(spanBase, makeChunk(context, bytes)).first->second;
}

// Records the upload of the blocks of `chunk` that [from, to) touches and that changed since their
// last upload (all of them when `allChanged`), from `import`. With `mayRefuse`, an upload that would
// end an open render pass for frequently rewritten blocks is not made and false is returned.
bool syncBlocks(Cache& state, const Context& context, Recorder& recorder, Chunk& chunk, std::uint64_t chunkBase, std::uint64_t from, std::uint64_t to, const HostImport& import, std::uint64_t generation, bool allChanged, bool mayRefuse, bool refuseRewritten = false) {
    const auto firstBlock = static_cast<std::size_t>((from - chunkBase) / BlockBytes);
    const auto lastBlock = static_cast<std::size_t>((to - 1 - chunkBase) / BlockBytes);
    const auto blocks = lastBlock - firstBlock + 1;
    thread_local std::vector<std::uint8_t> changed;
    changed.assign(blocks, 0);
    const auto blockBase = chunkBase + firstBlock * BlockBytes;
    GuestMemory::ChangedBlocks(blockBase, static_cast<std::size_t>(blocks * BlockBytes), std::span<const std::uint64_t>(chunk.uploaded.data() + firstBlock, blocks), std::span<std::uint8_t>(changed.data(), blocks));
    bool dirty = false;
    bool worth = true;
    for (std::size_t k = 0; k < blocks; ++k) {
        const auto block = firstBlock + k;
        if (chunk.uses[block] != std::numeric_limits<std::uint16_t>::max()) ++chunk.uses[block];
        if (allChanged) changed[k] = GuestMemory::BlockWritten;
        if (changed[k] == GuestMemory::BlockUnchanged) continue;
        dirty = true;
        worth = worth && worthPassBreak(chunk, block);
    }
    if (!dirty) {
        ++state.stats.clean;
        return true;
    }
    const bool passOpen = recorder.RenderPassOpen();
    if (mayRefuse && passOpen && !worth) {
        ++state.stats.refusedInPass;
        return false;
    }
    // Rewritten about every use (GPU-built lists in host memory): not worth a copy at all.
    if (refuseRewritten && !worth) {
        ++state.stats.refusedRewritten;
        return false;
    }
    // The changed blocks, clipped to the import (a block partly outside it stays marked as never
    // uploaded, so the next use copies it again from the import that holds it).
    std::vector<VkBufferCopy> copies;
    std::uint64_t copied = 0;
    const auto importEnd = import.base + import.bytes;
    for (std::size_t k = 0; k < blocks; ++k) {
        if (changed[k] == GuestMemory::BlockUnchanged) continue;
        const auto block = firstBlock + k;
        const auto begin = chunkBase + block * BlockBytes;
        const auto stop = begin + BlockBytes;
        const auto first = std::max(begin, import.base);
        const auto last = std::min(stop, importEnd);
        if (first >= last) {
            chunk.uploaded[block] = 0;
            continue;
        }
        const VkBufferCopy region{HostImportOffset(import, first), first - chunkBase, last - first};
        if (!copies.empty() && copies.back().srcOffset + copies.back().size == region.srcOffset && copies.back().dstOffset + copies.back().size == region.dstOffset) copies.back().size += region.size;
        else copies.push_back(region);
        copied += last - first;
        chunk.uploaded[block] = first == begin && last == stop ? generation : 0;
        if (chunk.uploads[block] != std::numeric_limits<std::uint16_t>::max()) ++chunk.uploads[block];
    }
    if (copies.empty()) return true;
    const auto commands = recorder.Commands();
    // Earlier work's writes into the import (in queue order) before the copy reads it, and earlier
    // reads of the chunk before it is overwritten.
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(commands, import.buffer, chunk.buffer->Handle(), static_cast<std::uint32_t>(copies.size()), copies.data());
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    ++state.stats.uploaded;
    state.stats.uploadBytes += copied;
    if (passOpen) ++state.stats.passBreaks;
    return true;
}

// The table's root (creating it zeroed on first use): a device-local buffer of BufferCacheTableRoots
// device addresses of leaves.
Buffer* tableRoot(Cache& state, const Context& context, Recorder& recorder) {
    if (state.root != nullptr) return state.root.get();
    if (!context.bufferDeviceAddress) return nullptr;
    constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    state.root = std::make_shared<Buffer>(context, static_cast<std::size_t>(BufferCacheTableRoots * sizeof(std::uint64_t)), usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const auto commands = recorder.Commands();
    context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, state.root->Handle(), 0, VK_WHOLE_SIZE, 0);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    return state.root.get();
}

// Writes the table entries of a chunk's pages (creating its leaf if needed). Entries are written
// with vkCmdUpdateBuffer in queue order, so work recorded after this call sees them.
bool mapChunk(Cache& state, const Context& context, Recorder& recorder, Chunk& chunk, std::uint64_t chunkBase, std::uint64_t from, std::uint64_t to) {
    const auto firstPage = static_cast<std::size_t>((from - chunkBase + TablePageBytes - 1) / TablePageBytes);
    const auto lastPage = static_cast<std::size_t>((to - chunkBase) / TablePageBytes);
    if (firstPage >= lastPage) return false;
    if (std::all_of(chunk.mapped.begin() + firstPage, chunk.mapped.begin() + lastPage, [](bool mapped) { return mapped; })) return true;
    auto* root = tableRoot(state, context, recorder);
    if (root == nullptr || chunk.buffer->DeviceAddress() == 0) return false;
    const auto rootIndex = chunkBase >> BufferCacheTableLeafBits;
    if (rootIndex >= BufferCacheTableRoots) return false;
    const auto update = context.Function<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer");
    auto& leaf = state.leaves[rootIndex];
    const auto commands = recorder.Commands();
    if (leaf == nullptr) {
        constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        leaf = std::make_shared<Buffer>(context, static_cast<std::size_t>(LeafEntries * sizeof(std::uint64_t)), usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, leaf->Handle(), 0, VK_WHOLE_SIZE, 0);
        const std::uint64_t leafAddress = leaf->DeviceAddress();
        update(commands, root->Handle(), rootIndex * sizeof(std::uint64_t), sizeof(leafAddress), &leafAddress);
    }
    std::array<std::uint64_t, ChunkBytes / TablePageBytes> entries{};
    for (std::size_t page = firstPage; page < lastPage; ++page) entries[page] = chunk.buffer->DeviceAddress() + page * TablePageBytes;
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    const auto leafOffset = (((chunkBase & (LeafSpan - 1)) / TablePageBytes) + firstPage) * sizeof(std::uint64_t);
    update(commands, leaf->Handle(), leafOffset, (lastPage - firstPage) * sizeof(std::uint64_t), entries.data() + firstPage);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    std::fill(chunk.mapped.begin() + firstPage, chunk.mapped.begin() + lastPage, true);
    return true;
}

}

int BufferCacheMode() {
    static const int mode = [] {
        const char* text = std::getenv("APS5_BUFFER_CACHE");
        return text != nullptr ? std::atoi(text) : 0;
    }();
    return mode;
}

bool BufferCacheEnabled() {
    return BufferCacheMode() > 0;
}

std::optional<CachedBufferRange> ObtainCachedBuffer(const Context& context, Recorder& recorder, std::uint64_t address, std::size_t bytes, bool* behindGpuWrite, bool mayRefuse, bool refuseRewritten) {
    if (!BufferCacheEnabled() || bytes == 0 || bytes > SpanMaxBytes || address > std::numeric_limits<std::uint64_t>::max() - bytes) return std::nullopt;
    auto& state = cache();
    const auto end = address + bytes;
    const bool crossing = (end - 1) / ChunkBytes != address / ChunkBytes;
    // Mode 2: data GPU work wrote into a resident staging copy (compute-built index lists, skinned
    // vertices) is read there, as KytyPS5 reads GPU-written buffers from their cache buffer: no copy
    // back to guest memory and up again. The caller's barrier orders the read after the writers.
    if (BufferCacheMode() >= 2) {
        if (auto resident = ResidentDrawSource(context.device, address, bytes)) {
            recorder.Keep(resident->keep, 0);
            recorder.NotePendingRead(address, bytes, Recorder::ReadKind::DrawInput);
            std::lock_guard lock(state.mutex);
            ++state.stats.obtained;
            ++state.stats.residentSourced;
            report(state);
            if (behindGpuWrite != nullptr) *behindGpuWrite = true;
            return CachedBufferRange{resident->buffer, resident->offset, true};
        }
    }
    // The checks and flushes of an in-place read: pending image results and resident copies over
    // the range go to guest memory (recorded), queued labels and copied writers refuse, and the
    // read is noted. The import is the upload source.
    bool behind = false;
    const auto* import = InPlaceDrawInput(context, address, bytes, 1, &behind, true);
    std::lock_guard lock(state.mutex);
    ++state.stats.obtained;
    const auto refuse = [&](std::uint64_t Stats::* counter) -> std::optional<CachedBufferRange> {
        if (counter != nullptr) ++(state.stats.*counter);
        report(state);
        return std::nullopt;
    };
    if (import == nullptr) return refuse(&Stats::noImport);
    const auto generation = GuestMemory::CollectWrites(address, bytes);
    if (generation == 0) return refuse(&Stats::untracked);
    resetFor(state, context.device);
    std::uint64_t chunkBase = address / ChunkBytes * ChunkBytes;
    auto* chunk = crossing ? spanFor(state, context, recorder, address, end, chunkBase) : chunkFor(state, context, chunkBase);
    if (chunk == nullptr) return refuse(nullptr);
    // Recorded GPU work writing the range stamped its blocks when it was recorded, but a resident
    // copy-back or image store the flush above just recorded may not have: such a range is copied.
    if (!syncBlocks(state, context, recorder, *chunk, chunkBase, address, end, *import, generation, behind, mayRefuse, refuseRewritten)) return refuse(nullptr);
    // The upload (if any) was recorded after every recorded writer, behind its barriers.
    if (behindGpuWrite != nullptr) *behindGpuWrite = false;
    report(state);
    return CachedBufferRange{chunk->buffer->Handle(), address - chunkBase};
}

VkDeviceAddress BufferCacheTable(const Context& context, Recorder& recorder) {
    auto& state = cache();
    std::lock_guard lock(state.mutex);
    resetFor(state, context.device);
    auto* root = tableRoot(state, context, recorder);
    return root != nullptr ? root->DeviceAddress() : 0;
}

namespace {
void noteResidentRange(Cache& state, std::uint64_t begin, std::uint64_t end) {
    auto& ranges = state.residentRanges;
    auto it = ranges.upper_bound(begin);
    if (it != ranges.begin() && std::prev(it)->second >= begin) --it;
    while (it != ranges.end() && it->first <= end) {
        begin = std::min(begin, it->first);
        end = std::max(end, it->second);
        it = ranges.erase(it);
    }
    ranges.emplace(begin, end);
}
}

bool MakeResident(const Context& context, Recorder& recorder, std::uint64_t address, std::size_t bytes) {
    if (!BufferCacheEnabled() || bytes == 0 || address > std::numeric_limits<std::uint64_t>::max() - bytes) return false;
    auto& state = cache();
    const auto end = address + bytes;
    bool whole = true;
    for (auto chunkBase = address / ChunkBytes * ChunkBytes; chunkBase < end; chunkBase += ChunkBytes) {
        const auto from = std::max(address, chunkBase);
        const auto to = std::min(end, chunkBase + ChunkBytes);
        bool behind = false;
        const auto* import = InPlaceDrawInput(context, from, static_cast<std::size_t>(to - from), 1, &behind, true);
        if (import == nullptr && to - from > TablePageBytes) {
            // No one import holds the whole part (two allocations meet in it): page by page.
            for (auto page = from / TablePageBytes * TablePageBytes; page < to; page += TablePageBytes) {
                const auto first = std::max(page, from);
                const auto last = std::min(page + TablePageBytes, to);
                whole = MakeResident(context, recorder, first, static_cast<std::size_t>(last - first)) && whole;
            }
            continue;
        }
        std::lock_guard lock(state.mutex);
        ++state.stats.resident;
        const auto generation = import != nullptr ? GuestMemory::CollectWrites(from, static_cast<std::size_t>(to - from)) : 0;
        resetFor(state, context.device);
        auto* chunk = import != nullptr && generation != 0 ? chunkFor(state, context, chunkBase) : nullptr;
        if (chunk == nullptr || !syncBlocks(state, context, recorder, *chunk, chunkBase, from, to, *import, generation, behind, false) || !mapChunk(state, context, recorder, *chunk, chunkBase, from, to)) {
            ++state.stats.residentRefused;
            whole = false;
        } else {
            noteResidentRange(state, from, to);
        }
        report(state);
    }
    return whole;
}

namespace {
struct ResidentQueue {
    std::mutex mutex;
    std::vector<std::pair<std::uint64_t, std::size_t>> ranges;
};
ResidentQueue& residentQueue() {
    static auto* queue = new ResidentQueue();
    return *queue;
}
}

void QueueResident(std::uint64_t address, std::size_t bytes) {
    if (!BufferCacheEnabled() || bytes == 0) return;
    auto& queue = residentQueue();
    std::lock_guard lock(queue.mutex);
    if (queue.ranges.size() < 65536) queue.ranges.emplace_back(address, bytes);
}

void MakeQueuedResident(const Context& context, Recorder& recorder) {
    if (!BufferCacheEnabled()) return;
    std::vector<std::pair<std::uint64_t, std::size_t>> ranges;
    {
        auto& queue = residentQueue();
        std::lock_guard lock(queue.mutex);
        if (queue.ranges.empty()) return;
        ranges.swap(queue.ranges);
    }
    std::sort(ranges.begin(), ranges.end());
    ranges.erase(std::unique(ranges.begin(), ranges.end()), ranges.end());
    for (const auto& [address, bytes] : ranges) static_cast<void>(MakeResident(context, recorder, address, bytes));
}

void SyncResident(const Context& context, Recorder& recorder) {
    if (!BufferCacheEnabled()) return;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    {
        auto& state = cache();
        std::lock_guard lock(state.mutex);
        if (state.device != context.device) return;
        ranges.assign(state.residentRanges.begin(), state.residentRanges.end());
    }
    for (const auto& [begin, end] : ranges) static_cast<void>(MakeResident(context, recorder, begin, static_cast<std::size_t>(end - begin)));
}

void ClearBufferCache(VkDevice device) {
    auto& state = cache();
    std::lock_guard lock(state.mutex);
    if (state.device == device) {
        state.chunks.clear();
        state.leaves.clear();
        state.root.reset();
    }
}

}
