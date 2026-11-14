#include "prx/libSceAgcDriver/Graphics/include/BindlessTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "BdaAbi.hpp"
#include "GlobalImageAbi.hpp"
#include "ImageTableAbi.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <vector>

namespace AgcDriver::Graphics {

std::shared_ptr<Texture> BindlessSampledTexture(const Context& context, std::span<const std::uint32_t, 8> words, ShaderRecompiler::DescriptorImageShape shape, bool depthCompare);
std::pair<std::uint64_t, std::uint64_t> BindlessTextureRange(std::span<const std::uint32_t, 8> words, ShaderRecompiler::DescriptorImageShape shape);

namespace {

namespace Abi = ShaderRecompiler::GlobalImageAbi;
using Entry = std::array<std::uint32_t, Abi::EntryWords>;
using Key = std::array<std::uint32_t, Abi::EntryKeyWords>;

constexpr std::uint32_t TombstoneKind = 2u;
constexpr std::uint32_t NoEntry = 0xffffffffu;
constexpr std::uint32_t ImageBindings = Abi::ViewBindings;
constexpr std::uint32_t MaxPendingAttempts = 8;

struct ImageSlot {
    std::shared_ptr<Texture> texture;
    std::uint32_t entry = NoEntry;
    std::uint32_t binding = 0;
    std::array<std::uint32_t, 8> words{};
    ShaderRecompiler::DescriptorImageShape shape = ShaderRecompiler::DescriptorImageShape::Image2D;
    bool depthCompare = false;
    // The texels' guest range and its write generation at the last lookup (0: not write-watched).
    std::uint64_t address = 0;
    std::uint64_t bytes = 0;
    std::uint64_t generation = 0;
    // Presents (Recorder::Presents) at the slot's last GPU use and at its last texture-cache touch.
    std::uint64_t usedPresent = 0;
    std::uint64_t touchedPresent = 0;
};

// A slot no shader sampled for this many presents is removed (its texture no longer pinned): 240
// kept ~600 textures and their storage images alive and the storage cache budget fell to its floor
// (eva_u32: ~2000 makes and releases per 10 s once it did).
constexpr std::uint64_t UnusedPresents = 60;
static_assert(ShaderRecompiler::GlobalImageAbi::MaxImageSlots == BindlessImageSlots && ShaderRecompiler::BdaAbi::GlobalImageUsedWords * 32u == BindlessImageSlots);

struct Miss {
    std::uint32_t kind = 0;
    Key key{};
    std::uint32_t attempts = 0;
};

struct PendingMisses {
    std::mutex mutex;
    std::vector<Miss> misses;
    // Image slots shaders sampled since the last flush (BdaAbi::ForEachUsedGlobalImageSlot).
    std::vector<std::uint32_t> used;
    // APS5_GPU_SRT_GLOBAL_COUNT totals (image hit/miss, sampler hit/miss) and misses received.
    std::array<std::uint64_t, ShaderRecompiler::BdaAbi::GlobalImageCountWords> counts{};
    std::uint64_t missesReceived = 0;
};

// Misses are kept per device: a key a shader of one device missed must be made in that device's set.
PendingMisses& pendingMisses(VkDevice device) {
    static std::mutex mutex;
    static auto* pending = new std::map<VkDevice, std::unique_ptr<PendingMisses>>();
    std::lock_guard lock(mutex);
    auto& slot = (*pending)[device];
    if (slot == nullptr) slot = std::make_unique<PendingMisses>();
    return *slot;
}

struct Stats {
    std::uint64_t updated = 0, approximated = 0, imagesMade = 0, samplersMade = 0, poisoned = 0, unknownKind = 0, refusedProbe = 0, refusedFull = 0, refreshed = 0, released = 0, perUse = 0, releasedObject = 0, releasedUnused = 0, releasedStale = 0, pendingRelooked = 0, missedPresent = 0, releasedRelooked = 0;
};

struct Table {
    std::recursive_mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    bool failed = false;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::shared_ptr<Buffer> map;
    bool mapInitialized = false;
    std::vector<Entry> entries;
    std::vector<ImageSlot> imageSlots;
    std::vector<std::uint32_t> freeImageSlots;
    std::deque<std::pair<std::uint64_t, std::uint32_t>> quarantine;
    std::vector<std::shared_ptr<Sampler>> samplers;
    Stats stats;
    std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point refreshed{};
    // Slots whose texture follows GPU-written content (a view of a storage or depth image, or a copy
    // the cache refreshes or remakes per use): rebuilt when a slot's descriptor changes.
    std::vector<std::uint32_t> gpuSlots;
    bool gpuSlotsDirty = true;
    // Snapshot slots (not GPU-backed) sampled in the last presents, rebuilt once per present, and
    // the storage registry's PendingSerial when they were last checked against pending GPU results.
    std::vector<std::uint32_t> recentSnapshots;
    std::uint64_t recentPresent = ~std::uint64_t{0};
    std::uint64_t pendingSerial = ~std::uint64_t{0};
};

// One table per live Vulkan device (the driver can hold more than one; their sets are not
// interchangeable). Tables live as long as the process.
Table& table(VkDevice device) {
    static std::mutex mutex;
    static auto* tables = new std::map<VkDevice, std::unique_ptr<Table>>();
    std::lock_guard lock(mutex);
    auto& slot = (*tables)[device];
    if (slot == nullptr) slot = std::make_unique<Table>();
    return *slot;
}

VkDeviceSize mapBytes() {
    return static_cast<VkDeviceSize>(Abi::HeaderWords + BindlessMapEntries * Abi::EntryWords) * sizeof(std::uint32_t);
}

bool create(Table& state, const Context& context) {
    if (state.device == context.device) return !state.failed;
    state.device = context.device;
    if (!context.descriptorIndexing || !context.partiallyBound || !context.runtimeDescriptorArray || !context.sampledImageUpdateAfterBind) {
        state.failed = true;
        return false;
    }
    std::array<VkDescriptorSetLayoutBinding, ImageBindings + 2> bindings{};
    std::array<VkDescriptorBindingFlags, ImageBindings + 2> flags{};
    for (std::uint32_t binding = 0; binding < ImageBindings; ++binding) {
        bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, BindlessImageSlots, VK_SHADER_STAGE_ALL, nullptr};
        flags[binding] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    }
    bindings[Abi::SamplerBinding] = {Abi::SamplerBinding, VK_DESCRIPTOR_TYPE_SAMPLER, BindlessSamplerSlots, VK_SHADER_STAGE_ALL, nullptr};
    flags[Abi::SamplerBinding] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    bindings[Abi::MapBinding] = {Abi::MapBinding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr};
    static_assert(Abi::SamplerBinding == ImageBindings && Abi::MapBinding == ImageBindings + 1);
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bindingFlags.bindingCount = static_cast<std::uint32_t>(flags.size());
    bindingFlags.pBindingFlags = flags.data();
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &bindingFlags};
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &layoutInfo, nullptr, &state.layout) != VK_SUCCESS) {
        state.failed = true;
        return false;
    }
    const std::array<VkDescriptorPoolSize, 3> sizes{{{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, BindlessImageSlots * ImageBindings}, {VK_DESCRIPTOR_TYPE_SAMPLER, BindlessSamplerSlots}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}}};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    poolInfo.pPoolSizes = sizes.data();
    if (context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &state.pool) != VK_SUCCESS) {
        state.failed = true;
        return false;
    }
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = state.pool;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &state.layout;
    if (context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &state.set) != VK_SUCCESS) {
        state.failed = true;
        return false;
    }
    constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    state.map = std::make_shared<Buffer>(context, mapBytes(), usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    state.entries.assign(BindlessMapEntries, Entry{});
    state.imageSlots.resize(BindlessImageSlots);
    for (std::uint32_t slot = BindlessImageSlots; slot-- > 0;) state.freeImageSlots.push_back(slot);
    state.samplers.reserve(BindlessSamplerSlots);
    const VkDescriptorBufferInfo mapInfo{state.map->Handle(), 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = state.set;
    write.dstBinding = Abi::MapBinding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &mapInfo;
    context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 1, &write, 0, nullptr);
    return true;
}

class MapWriter {
public:
    MapWriter(Table& state, const Context& context, Recorder& recorder) : state(state), context(context), recorder(recorder) {}
    ~MapWriter() {
        if (!open) return;
        RecordMemoryBarrier(context, recorder.Commands(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    void Begin() {
        if (open) return;
        open = true;
        RecordMemoryBarrier(context, recorder.Commands(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    }
    void Initialize() {
        if (state.mapInitialized) return;
        Begin();
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(recorder.Commands(), state.map->Handle(), 0, VK_WHOLE_SIZE, 0u);
        RecordMemoryBarrier(context, recorder.Commands(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        const std::array<std::uint32_t, Abi::HeaderWords> header{Abi::Version, BindlessMapEntries, 0u, 0u};
        context.Function<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer")(recorder.Commands(), state.map->Handle(), 0, sizeof(header), header.data());
        state.mapInitialized = true;
    }
    void Write(std::uint32_t index) {
        Begin();
        if (std::find(written.begin(), written.end(), index) != written.end()) {
            RecordMemoryBarrier(context, recorder.Commands(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            written.clear();
        }
        const auto offset = static_cast<VkDeviceSize>(Abi::EntryWord(index, 0)) * sizeof(std::uint32_t);
        context.Function<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer")(recorder.Commands(), state.map->Handle(), offset, sizeof(Entry), state.entries[index].data());
        written.push_back(index);
    }

private:
    Table& state;
    const Context& context;
    Recorder& recorder;
    bool open = false;
    std::vector<std::uint32_t> written;
};

std::uint32_t probe(const Table& state, std::uint32_t kind, const Key& key, bool& found) {
    found = false;
    const auto mask = BindlessMapEntries - 1u;
    const auto start = Abi::Hash(key, kind) & mask;
    std::uint32_t free = NoEntry;
    for (std::uint32_t k = 0; k < Abi::Probes; ++k) {
        const auto index = (start + k) & mask;
        const auto& entry = state.entries[index];
        const auto stored = entry[Abi::EntryKind];
        if (stored == Abi::KindEmpty) return free != NoEntry ? free : index;
        if (stored == TombstoneKind) {
            if (free == NoEntry) free = index;
            continue;
        }
        if (stored == kind && std::equal(key.begin(), key.end(), entry.begin() + Abi::EntryKey)) {
            found = true;
            return index;
        }
    }
    return free;
}

// A texture whose content follows GPU work: the direct path looks it up on every draw (the lookup
// refreshes a per-use copy, remakes a per-use object, and the draw lists the image it reads so the
// recorder orders it after the pass that wrote it). Views of storage or depth images are kept in the
// general layout; snapshots are not.
bool gpuBacked(const Texture& texture) {
    return texture.Layout() == VK_IMAGE_LAYOUT_GENERAL || texture.CopyKind() != Texture::PerUseCopy::None;
}

// A slot sampled in this or the previous present (or just made).
bool recentlyUsed(const ImageSlot& kept, std::uint64_t present) {
    return std::max(kept.usedPresent, kept.touchedPresent) + 2u >= present;
}

void writeImageDescriptor(Table& state, const Context& context, std::uint32_t slot) {
    state.gpuSlotsDirty = true;
    const auto& kept = state.imageSlots[slot];
    const VkDescriptorImageInfo image{VK_NULL_HANDLE, kept.texture->View(), kept.texture->Layout()};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = state.set;
    write.dstBinding = kept.binding;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &image;
    context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 1, &write, 0, nullptr);
}

bool make(Table& state, const Context& context, MapWriter& writer, const Miss& miss) {
    bool found = false;
    const auto index = probe(state, miss.kind, miss.key, found);
    if (found) {
        // The GPU missed a key the map holds (written after the batch's lookups, or not found by
        // the shader's probe): counted.
        ++state.stats.missedPresent;
        return true;
    }
    if (index == NoEntry) {
        ++state.stats.refusedProbe;
        return true;
    }
    ShaderRecompiler::GlobalImageKeyResolution resolution;
    if (!ShaderRecompiler::ResolveGlobalImageKey(miss.kind, std::span<const std::uint32_t, 8>(miss.key), resolution)) {
        ++state.stats.unknownKind;
        return false;
    }
    Entry entry{};
    std::copy(miss.key.begin(), miss.key.end(), entry.begin() + Abi::EntryKey);
    entry[Abi::EntryKind] = miss.kind;
    entry[Abi::EntryFlags] = resolution.flags;
    const bool poisoned = (resolution.flags & ShaderRecompiler::ImageTableAbi::PoisonFlag) != 0u;
    if (poisoned) {
        ++state.stats.poisoned;
    } else if (resolution.sampler) {
        if (state.samplers.size() + ShaderRecompiler::ImageTableAbi::SamplerEntryElements > BindlessSamplerSlots || context.samplerCache == nullptr) {
            ++state.stats.refusedFull;
            return true;
        }
        const auto first = static_cast<std::uint32_t>(state.samplers.size());
        std::array<VkDescriptorImageInfo, ShaderRecompiler::ImageTableAbi::SamplerEntryElements> infos{};
        // Sampler states the direct path pairs with each image (shader-applied LOD bias, border
        // colors of integer images, degamma, non-seamless cubes, ANISO_OVERRIDE) are approximated by
        // the plain sampler (counted); one the device cannot make poisons the entry (samples 0)
        // instead of failing the draw.
        std::vector<std::shared_ptr<Sampler>> made;
        bool usable = true;
        for (std::uint32_t variant = 0; variant < infos.size() && usable; ++variant) {
            const auto words = std::span<const std::uint32_t, 4>(resolution.samplerWords[variant]);
            try {
                auto resource = DecodeSamplerResource(words);
                resource.compareEnable = variant >= 2u;
                Sampler::Validate(context, resource);
                const bool approximated = Sampler::ShaderAppliesLodBias(resource) || resource.disableDegamma || (resource.nonSeamlessCube && !context.nonSeamlessCubeMap) || SingleLevelSamplerWords(words, true, false).has_value();
                if (approximated && variant == 0u) ++state.stats.approximated;
                auto sampler = context.samplerCache->Get(context, resolution.samplerWords[variant], variant >= 2u, false);
                infos[variant] = {sampler->Handle(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
                made.push_back(std::move(sampler));
            } catch (const std::exception& error) {
                static std::atomic<std::uint32_t> reported{0};
                if (reported.fetch_add(1, std::memory_order_relaxed) < 20u) std::fprintf(stderr, "[bindless] S# %08x %08x %08x %08x poisoned: %s\n", words[0], words[1], words[2], words[3], error.what());
                usable = false;
            }
        }
        if (!usable) {
            entry[Abi::EntryFlags] = ShaderRecompiler::ImageTableAbi::PoisonFlag;
            ++state.stats.poisoned;
            state.entries[index] = entry;
            writer.Write(index);
            return true;
        }
        for (auto& sampler : made) state.samplers.push_back(std::move(sampler));
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = state.set;
        write.dstBinding = Abi::SamplerBinding;
        write.dstArrayElement = first;
        write.descriptorCount = static_cast<std::uint32_t>(infos.size());
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        write.pImageInfo = infos.data();
        context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 1, &write, 0, nullptr);
        entry[Abi::EntrySamplerElement] = first;
        ++state.stats.samplersMade;
    } else {
        std::shared_ptr<Texture> texture;
        try {
            texture = BindlessSampledTexture(context, std::span<const std::uint32_t, 8>(resolution.words), resolution.shape, resolution.depthCompare);
        } catch (const std::exception& error) {
            // A texture the cache refuses (a state the direct path pairs per draw) samples 0.
            static std::atomic<std::uint32_t> reported{0};
            if (reported.fetch_add(1, std::memory_order_relaxed) < 20u) std::fprintf(stderr, "[bindless] T# %08x %08x %08x %08x poisoned: %s\n", resolution.words[0], resolution.words[1], resolution.words[2], resolution.words[3], error.what());
        }
        if (texture == nullptr) {
            entry[Abi::EntryFlags] = ShaderRecompiler::ImageTableAbi::PoisonFlag;
            ++state.stats.poisoned;
        } else {
            if (state.freeImageSlots.empty()) {
                ++state.stats.refusedFull;
                return true;
            }
            const auto slot = state.freeImageSlots.back();
            state.freeImageSlots.pop_back();
            state.imageSlots[slot] = {std::move(texture), index, resolution.viewBinding, resolution.words, resolution.shape, resolution.depthCompare};
            state.imageSlots[slot].touchedPresent = Recorder::Presents();
            const auto [address, bytes] = BindlessTextureRange(std::span<const std::uint32_t, 8>(resolution.words), resolution.shape);
            state.imageSlots[slot].address = address;
            state.imageSlots[slot].bytes = bytes;
            state.imageSlots[slot].generation = bytes != 0 ? GuestMemory::CollectWrites(address, static_cast<std::size_t>(bytes)) : 0;
            writeImageDescriptor(state, context, slot);
            entry[Abi::EntryDword3] = resolution.dword3;
            entry[Abi::EntryElement] = slot;
            ++state.stats.imagesMade;
        }
    }
    state.entries[index] = entry;
    writer.Write(index);
    return true;
}

// Moves slot's entry to a fresh element viewing current (the old element is reused once the batches
// that may still read it completed).
void replaceSlot(Table& state, const Context& context, Recorder& recorder, MapWriter& writer, std::uint32_t slot, std::shared_ptr<Texture> current, std::uint64_t generation) {
    auto& kept = state.imageSlots[slot];
    const auto replacement = state.freeImageSlots.back();
    state.freeImageSlots.pop_back();
    state.imageSlots[replacement] = kept;
    state.imageSlots[replacement].texture = std::move(current);
    state.imageSlots[replacement].generation = generation;
    writeImageDescriptor(state, context, replacement);
    state.entries[kept.entry][Abi::EntryElement] = replacement;
    writer.Write(kept.entry);
    kept = {};
    state.quarantine.emplace_back(recorder.OpenSerial(), slot);
    ++state.stats.refreshed;
}

// Before each draw or dispatch with global images: what the direct path's per-draw lookup does for
// GPU-backed textures sampled recently: a depth/block/sRGB copy is brought up to its source's
// version, an object made per use is looked up again (and the slot moved to it when it changed).
// Without this the global set kept the copy made at the first miss (eva_u27/u30: black frames).
void refreshGpuBacked(Table& state, const Context& context, Recorder& recorder, MapWriter& writer) {
    const auto present = Recorder::Presents();
    if (state.gpuSlotsDirty) {
        state.gpuSlots.clear();
        for (std::uint32_t slot = 0; slot < state.imageSlots.size(); ++slot) {
            const auto& texture = state.imageSlots[slot].texture;
            if (texture != nullptr && gpuBacked(*texture)) state.gpuSlots.push_back(slot);
        }
        state.gpuSlotsDirty = false;
    }
    const auto slots = state.gpuSlots;
    for (const auto slot : slots) {
        auto& kept = state.imageSlots[slot];
        if (kept.texture == nullptr || kept.texture->Released() || !recentlyUsed(kept, present)) continue;
        if (kept.texture->IsDepthCopy()) {
            if (kept.texture->RefreshDepthCopy()) ++state.stats.perUse;
            continue;
        }
        if (!kept.texture->RefreshedPerUse()) continue;
        std::shared_ptr<Texture> current;
        try {
            current = BindlessSampledTexture(context, std::span<const std::uint32_t, 8>(kept.words), kept.shape, kept.depthCompare);
        } catch (const std::exception&) {
            continue;
        }
        ++state.stats.perUse;
        if (current == nullptr || current == kept.texture || state.freeImageSlots.empty()) continue;
        recorder.Keep(kept.texture, 0);
        replaceSlot(state, context, recorder, writer, slot, std::move(current), kept.generation);
    }
}

// A snapshot slot whose memory now has GPU results pending (a render target or storage image the
// GPU wrote after the slot was made: an exposure value, a lit buffer) is looked up again, as the
// direct path does on every draw: the cache then serves a view of (or a copy from) that image, and
// the slot becomes GPU-backed. The check runs when the pending registry changed (PendingSerial),
// one registry scan for all recent snapshot ranges first.
void refreshPendingSnapshots(Table& state, const Context& context, Recorder& recorder, MapWriter& writer) {
    const auto present = Recorder::Presents();
    if (state.recentPresent != present) {
        state.recentPresent = present;
        state.recentSnapshots.clear();
        for (std::uint32_t slot = 0; slot < state.imageSlots.size(); ++slot) {
            const auto& kept = state.imageSlots[slot];
            if (kept.texture != nullptr && kept.bytes != 0 && !gpuBacked(*kept.texture) && recentlyUsed(kept, present)) state.recentSnapshots.push_back(slot);
        }
        state.pendingSerial = ~std::uint64_t{0};
    }
    const auto serial = StorageTexture::PendingSerial();
    if (serial == state.pendingSerial || state.recentSnapshots.empty()) return;
    state.pendingSerial = serial;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    ranges.reserve(state.recentSnapshots.size());
    for (const auto slot : state.recentSnapshots) {
        const auto& kept = state.imageSlots[slot];
        if (kept.texture != nullptr) ranges.emplace_back(kept.address, kept.address + kept.bytes);
    }
    if (!StorageTexture::AnyPendingOverlaps(ranges)) return;
    const auto slots = state.recentSnapshots;
    for (const auto slot : slots) {
        auto& kept = state.imageSlots[slot];
        if (kept.texture == nullptr || gpuBacked(*kept.texture)) continue;
        const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> range{std::pair{kept.address, kept.address + kept.bytes}};
        if (!StorageTexture::AnyPendingOverlaps(range)) continue;
        std::shared_ptr<Texture> current;
        try {
            current = BindlessSampledTexture(context, std::span<const std::uint32_t, 8>(kept.words), kept.shape, kept.depthCompare);
        } catch (const std::exception&) {
            continue;
        }
        ++state.stats.pendingRelooked;
        if (current == nullptr || current == kept.texture || state.freeImageSlots.empty()) continue;
        recorder.Keep(kept.texture, 0);
        replaceSlot(state, context, recorder, writer, slot, std::move(current), kept.generation);
    }
}

void refresh(Table& state, const Context& context, Recorder& recorder, MapWriter& writer) {
    while (!state.quarantine.empty() && state.quarantine.front().first <= recorder.CompletedThrough()) {
        state.freeImageSlots.push_back(state.quarantine.front().second);
        state.quarantine.pop_front();
    }
    std::vector<std::uint32_t> live;
    for (std::uint32_t slot = 0; slot < state.imageSlots.size(); ++slot) if (state.imageSlots[slot].texture != nullptr) live.push_back(slot);
    for (const auto slot : live) {
        auto& kept = state.imageSlots[slot];
        if (kept.texture == nullptr) continue;
        // A slot is looked up again only when its texture-cache object was released (replaced or
        // evicted) or the CPU wrote its texels since the last lookup (the cache brings the object up
        // to date or replaces it then): asking the cache for every live slot recreated GPU-written
        // textures on each walk (eva_m3f: ~9.6 GB of uploads per 10 s).
        std::uint64_t generation = kept.generation;
        // A released object (evicted by the texture cache budget, or replaced) is not made again here:
        // the entry is removed and the next miss of a shader that still uses the key makes it. Making
        // every released slot again kept ~1500 textures alive against a 2 GB budget and thrashed the
        // cache (eva_u26: ~800 re-creations per second, GPU ~500 ms per frame).
        const bool unused = kept.usedPresent != 0 ? Recorder::Presents() > kept.usedPresent + UnusedPresents : Recorder::Presents() > kept.touchedPresent + UnusedPresents;
        // A view of a storage image the storage cache let go no longer follows the surface (the
        // direct path's next lookup makes a new image): removed like a released object.
        const bool stale = kept.texture->StorageSource() != nullptr && !kept.texture->StorageSource()->Cached();
        // A released (or storage-less) object shaders still sample is replaced by the cache's current
        // object for the key instead of removed: a removed slot samples 0 until the next batch's miss
        // is made, and the cache releases ~230 objects a second (eva_u36), so a texture replaced every
        // frame was never sampled (APS5_BINDLESS_NO_RELOOK_RELEASED: remove them as before).
        static const bool relook = std::getenv("APS5_BINDLESS_NO_RELOOK_RELEASED") == nullptr;
        if (relook && !unused && (kept.texture->Released() || stale) && !state.freeImageSlots.empty()) {
            std::shared_ptr<Texture> current;
            try {
                current = BindlessSampledTexture(context, std::span<const std::uint32_t, 8>(kept.words), kept.shape, kept.depthCompare);
            } catch (const std::exception&) {
                current.reset();
            }
            if (current != nullptr && current != kept.texture && !current->Released()) {
                ++state.stats.releasedRelooked;
                recorder.Keep(kept.texture, 0);
                const auto refreshedGeneration = kept.bytes != 0 ? GuestMemory::CollectWrites(kept.address, static_cast<std::size_t>(kept.bytes)) : 0;
                replaceSlot(state, context, recorder, writer, slot, std::move(current), refreshedGeneration);
                continue;
            }
        }
        if (kept.texture->Released() || unused || stale) {
            ++(kept.texture->Released() ? state.stats.releasedObject : unused ? state.stats.releasedUnused : state.stats.releasedStale);
            recorder.Keep(kept.texture, 0);
            state.entries[kept.entry][Abi::EntryKind] = TombstoneKind;
            writer.Write(kept.entry);
            kept = {};
            state.quarantine.emplace_back(recorder.OpenSerial(), slot);
            ++state.stats.released;
            continue;
        }
        {
            if (kept.generation == 0 || kept.bytes == 0) continue;
            generation = GuestMemory::CollectWrites(kept.address, static_cast<std::size_t>(kept.bytes));
            if (generation == kept.generation || !GuestMemory::WrittenSince(kept.address, static_cast<std::size_t>(kept.bytes), kept.generation)) {
                kept.generation = generation;
                continue;
            }
        }
        std::shared_ptr<Texture> current;
        try {
            current = BindlessSampledTexture(context, std::span<const std::uint32_t, 8>(kept.words), kept.shape, kept.depthCompare);
        } catch (const std::exception&) {
            current.reset();  // removed below: the next miss resolves (and poisons) it again
        }
        if (current != nullptr && current == kept.texture) {
            kept.generation = generation;
            ++state.stats.updated;
            continue;
        }
        recorder.Keep(kept.texture, 0);
        if (current != nullptr) {
            Require(!state.freeImageSlots.empty(), "bindless image slots exhausted during refresh");
            replaceSlot(state, context, recorder, writer, slot, std::move(current), generation);
            continue;
        }
        state.entries[kept.entry][Abi::EntryKind] = TombstoneKind;
        writer.Write(kept.entry);
        kept = {};
        state.quarantine.emplace_back(recorder.OpenSerial(), slot);
        ++state.stats.released;
    }
}

void report(Table& state, PendingMisses& pending) {
    const auto now = std::chrono::steady_clock::now();
    if (now - state.reported < std::chrono::seconds(10)) return;
    state.reported = now;
    {
        std::lock_guard lock(pending.mutex);
        std::fprintf(stderr, "[bindless] (10 s) misses received %llu (key already mapped %llu); first-lane lookups: images %llu hit %llu missed, samplers %llu hit %llu missed\n", static_cast<unsigned long long>(pending.missesReceived), static_cast<unsigned long long>(state.stats.missedPresent), static_cast<unsigned long long>(pending.counts[0]), static_cast<unsigned long long>(pending.counts[1]), static_cast<unsigned long long>(pending.counts[2]), static_cast<unsigned long long>(pending.counts[3]));
        pending.counts = {};
        pending.missesReceived = 0;
    }
    const auto& s = state.stats;
    if (s.updated + s.imagesMade + s.samplersMade + s.poisoned + s.unknownKind + s.refusedProbe + s.refusedFull + s.refreshed + s.released == 0) return;
    std::fprintf(stderr, "[bindless] (10 s) texel updates %llu, images made %llu, samplers made %llu, poisoned %llu, unknown kind %llu, refreshed %llu, released %llu (object %llu, unused %llu, storage gone %llu; %llu looked up again instead), per-use refreshes %llu, snapshots over pending results looked up %llu, refused: probe run full %llu, slots full %llu; live image slots %zu (%zu GPU-backed), sampler slots %zu\n", static_cast<unsigned long long>(s.updated), static_cast<unsigned long long>(s.imagesMade), static_cast<unsigned long long>(s.samplersMade), static_cast<unsigned long long>(s.poisoned), static_cast<unsigned long long>(s.unknownKind), static_cast<unsigned long long>(s.refreshed), static_cast<unsigned long long>(s.released), static_cast<unsigned long long>(s.releasedObject), static_cast<unsigned long long>(s.releasedUnused), static_cast<unsigned long long>(s.releasedStale), static_cast<unsigned long long>(s.releasedRelooked), static_cast<unsigned long long>(s.perUse), static_cast<unsigned long long>(s.pendingRelooked), static_cast<unsigned long long>(s.refusedProbe), static_cast<unsigned long long>(s.refusedFull), BindlessImageSlots - state.freeImageSlots.size() - state.quarantine.size(), state.gpuSlots.size(), state.samplers.size());
    state.stats = {};
}

}

bool BindlessEnabled(const Context& context) {
    static const bool disabled = std::getenv("APS5_NO_BINDLESS") != nullptr || std::getenv("APS5_GPU_SRT") == nullptr || std::string_view(std::getenv("APS5_GPU_SRT")) != "4";
    if (disabled) return false;
    auto& state = table(context.device);
    std::lock_guard lock(state.mutex);
    return create(state, context);
}

VkDescriptorSetLayout BindlessSetLayout(const Context& context) {
    if (!BindlessEnabled(context)) return VK_NULL_HANDLE;
    return table(context.device).layout;
}

VkDescriptorSet BindlessSet(const Context& context) {
    if (!BindlessEnabled(context)) return VK_NULL_HANDLE;
    return table(context.device).set;
}

void QueueBindlessMisses(VkDevice device, std::uint32_t* faultWords) {
    if (std::getenv("APS5_GPU_SRT") == nullptr || std::string_view(std::getenv("APS5_GPU_SRT")) != "4") return;
    std::vector<std::uint32_t> used;
    ShaderRecompiler::BdaAbi::ForEachUsedGlobalImageSlot(faultWords, [&](std::uint32_t slot) { used.push_back(slot); });
    {
        std::array<std::uint32_t, ShaderRecompiler::BdaAbi::GlobalImageCountWords> counts{};
        bool any = false;
        for (std::uint32_t word = 0; word < counts.size(); ++word) {
            counts[word] = faultWords[ShaderRecompiler::BdaAbi::GlobalImageCountWord + word];
            faultWords[ShaderRecompiler::BdaAbi::GlobalImageCountWord + word] = 0;
            any = any || counts[word] != 0;
        }
        if (any) {
            auto& state = pendingMisses(device);
            std::lock_guard lock(state.mutex);
            for (std::uint32_t word = 0; word < counts.size(); ++word) state.counts[word] += counts[word];
        }
    }
    if (!used.empty()) {
        auto& state = pendingMisses(device);
        std::lock_guard lock(state.mutex);
        if (state.used.size() < 65536u) state.used.insert(state.used.end(), used.begin(), used.end());
    }
    std::vector<Miss> noted;
    ShaderRecompiler::BdaAbi::ForEachGlobalImageMiss(faultWords, [&](std::uint32_t kind, std::span<const std::uint32_t, 8> key) {
        Miss miss;
        miss.kind = kind;
        std::copy(key.begin(), key.end(), miss.key.begin());
        noted.push_back(miss);
    });
    if (noted.empty()) return;
    auto& state = pendingMisses(device);
    std::lock_guard lock(state.mutex);
    for (const auto& miss : noted) {
        if (state.misses.size() >= 4096u) break;
        if (std::ranges::any_of(state.misses, [&](const Miss& known) { return known.kind == miss.kind && known.key == miss.key; })) continue;
        state.misses.push_back(miss);
    }
}

void FlushBindless(const Context& context, Recorder& recorder) {
    auto& state = table(context.device);
    auto& pending = pendingMisses(context.device);
    std::vector<Miss> misses;
    std::vector<std::uint32_t> used;
    {
        std::lock_guard lock(pending.mutex);
        misses.swap(pending.misses);
        used.swap(pending.used);
        pending.missesReceived += misses.size();
    }
    {
        std::lock_guard lock(state.mutex);
        if (misses.empty() && used.empty() && (state.device != context.device || state.failed)) return;
    }
    if (!BindlessEnabled(context)) return;
    std::lock_guard lock(state.mutex);
    MapWriter writer(state, context, recorder);
    writer.Initialize();
    // Slots shaders sampled: their textures are marked used this present (Texture::NoteBindlessUse), so
    // the texture cache's eviction keeps them (it only saw CPU lookups as uses before: eva_u27 evicted
    // every sampled texture, frames came out black). A cache lookup per used slot instead (build 42)
    // flushed and waited inside FlushBindless and stalled the load.
    const auto present = Recorder::Presents();
    for (const auto slot : used) {
        if (slot >= state.imageSlots.size()) continue;
        auto& kept = state.imageSlots[slot];
        if (kept.texture == nullptr) continue;
        kept.usedPresent = present;
        kept.touchedPresent = present;
        kept.texture->NoteBindlessUse(present);
    }
    refreshGpuBacked(state, context, recorder, writer);
    refreshPendingSnapshots(state, context, recorder, writer);
    // The live-slot walk asks the texture cache for every slot's object: at most every 16 ms or when
    // keys are waiting (once per draw cost ~5 ms per draw with 23 slots in eva_m3e).
    const auto now = std::chrono::steady_clock::now();
    if (!misses.empty() || now - state.refreshed >= std::chrono::milliseconds(16)) {
        state.refreshed = now;
        refresh(state, context, recorder, writer);
    }
    for (auto& miss : misses) {
        if (make(state, context, writer, miss) || ++miss.attempts >= MaxPendingAttempts) continue;
        std::lock_guard pendingLock(pending.mutex);
        pending.misses.push_back(miss);
    }
    report(state, pending);
}

void BindlessDrawReads(const Context& context, std::vector<std::pair<VkImage, bool>>& images, std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges, std::span<const std::pair<std::uint64_t, std::uint64_t>> excluded) {
    if (!BindlessEnabled(context)) return;
    auto& state = table(context.device);
    std::lock_guard lock(state.mutex);
    const auto present = Recorder::Presents();
    const auto attachment = [&](VkImage image) { return std::ranges::any_of(images, [&](const auto& listed) { return listed.first == image && listed.second; }); };
    for (const auto slot : state.gpuSlots) {
        const auto& kept = state.imageSlots[slot];
        if (kept.texture == nullptr || !recentlyUsed(kept, present)) continue;
        if (const auto* source = kept.texture->StorageSource(); source != nullptr) {
            if (!attachment(source->Image())) images.emplace_back(source->Image(), false);
            continue;
        }
        if (kept.bytes == 0) continue;
        const auto end = kept.address + kept.bytes;
        if (std::ranges::any_of(excluded, [&](const auto& range) { return kept.address < range.second && range.first < end; })) continue;
        ranges.emplace_back(kept.address, end);
    }
}

}
