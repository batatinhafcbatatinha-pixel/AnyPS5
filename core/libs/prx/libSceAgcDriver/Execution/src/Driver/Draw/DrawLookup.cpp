#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <span>

namespace AgcDriver::DriverDetail {

namespace {

// Why the shape entry's stages could or could not serve a draw whose register key missed: a used
// user word changed that cannot be forwarded, the captured memory differs, served as kept, or
// served with changed forwarded words patched into a copy of the result
// (APS5_TRACE_DRAW_DATA_REUSE=1 prints the counts every 10 s).
struct DataReuseCounts {
    std::uint64_t userWords = 0, memory = 0, plain = 0, patched = 0;
};

void NoteDataReuse(const DataReuseCounts& counts) {
    static const bool enabled = std::getenv("APS5_TRACE_DRAW_DATA_REUSE") != nullptr;
    if (!enabled) return;
    static std::mutex mutex;
    static DataReuseCounts totals;
    static auto reported = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex);
    totals.userWords += counts.userWords;
    totals.memory += counts.memory;
    totals.plain += counts.plain;
    totals.patched += counts.patched;
    const auto now = std::chrono::steady_clock::now();
    if (now - reported < std::chrono::seconds(10)) return;
    reported = now;
    std::fprintf(stderr, "[draw-data-reuse] (10 s) shape-entry stage variants: refused for a changed user word %llu, captured memory differs %llu, served %llu, served with forwarded user words patched %llu\n", static_cast<unsigned long long>(totals.userWords), static_cast<unsigned long long>(totals.memory), static_cast<unsigned long long>(totals.plain), static_cast<unsigned long long>(totals.patched));
    totals = {};
}

// Changed user words a kept stage tolerates (APS5_DRAW_FORWARDED_HITS=1, meant for APS5_GPU_SRT=3,
// where the SRT pointers the shader reads itself are forwarded push data): each must be one its
// shader only reads at run time as forwarded push data (one slot of shaderDataUserWords), never
// walked by the capture (a base or a value of the SRT walk) nor a vertex fetch register; the slot
// is then patched as an indirect record's word is (patchForwardedWord).
bool ForwardedHits() {
    static const bool enabled = std::getenv("APS5_DRAW_FORWARDED_HITS") != nullptr;
    return enabled;
}

}

std::int64_t Driver::forwardedSlot(const DispatchVariant& variant, std::size_t word) {
    if (!ForwardedHits() || word >= 64u || variant.compiled == nullptr || variant.userDataMask == ~std::uint64_t{0} || ((variant.userWordsWalked >> word) & 1u) != 0) return -1;
    if (variant.vertexInfo != nullptr) {
        for (const auto reg : {static_cast<std::uint32_t>(variant.vertexInfo->fetchAttribReg), static_cast<std::uint32_t>(variant.vertexInfo->fetchBufferReg)}) {
            if (word == reg || word == static_cast<std::size_t>(reg) + 1u) return -1;
        }
    }
    const auto& slots = variant.compiled->shaderDataUserWords;
    const auto found = std::find(slots.begin(), slots.end(), static_cast<std::uint32_t>(word));
    if (found == slots.end() || std::find(found + 1, slots.end(), static_cast<std::uint32_t>(word)) != slots.end()) return -1;
    return found - slots.begin();
}

std::shared_ptr<DispatchVariant> Driver::forwardedCopy(const DispatchVariant& variant, std::span<const std::pair<std::uint32_t, std::uint32_t>> forwarded) {
    auto result = std::make_shared<ShaderRecompiler::RecompileResult>(*variant.compiled);
    for (const auto& [slot, value] : forwarded) patchForwardedWord(*result, slot, value);
    auto copy = std::make_shared<DispatchVariant>();
    copy->runs = variant.runs;
    copy->words = variant.words;
    copy->shared = variant.shared;
    copy->forgetSerial = variant.forgetSerial;
    copy->compiled = std::move(result);
    copy->registeredViews = variant.registeredViews;
    copy->meshIndexDescriptor = variant.meshIndexDescriptor;
    copy->shader = variant.shader;
    copy->dataPositions = variant.dataPositions;
    copy->dataSlots = variant.dataSlots;
    copy->flatBinding = variant.flatBinding;
    copy->pushOffset = variant.pushOffset;
    copy->runBases = variant.runBases;
    copy->relocatableWords = variant.relocatableWords;
    copy->userWordsWalked = variant.userWordsWalked;
    copy->valueWords = variant.valueWords;
    copy->valueSlots = variant.valueSlots;
    copy->calleeWords = variant.calleeWords;
    copy->calleeReads = variant.calleeReads;
    copy->userDataMask = variant.userDataMask;
    copy->vertexInfo = variant.vertexInfo;
    return copy;
}

void Driver::lookupDraw(const Submission& submission, const std::shared_ptr<VulkanDevice>& localDevice, const Graphics::State& graphics, const ShaderRecompiler::ShaderPixelStageInfo& pixel, const std::vector<DrawProgram>& programs, const std::vector<ShaderRecompiler::ProgramRole>& roles, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, bool useDrawEntries, bool registerKey, bool profile, std::uint64_t& drawKey, std::shared_ptr<DrawEntry>& entry, std::vector<std::shared_ptr<DispatchVariant>>& matched, std::vector<std::vector<ShaderRecompiler::MemoryRegion>>& matchedRegions, bool& drawHit, bool& verifyHit, DrawPhaseTiming& phaseTiming, std::array<double, DrawDriverPhaseCount>& phaseMs, const std::shared_ptr<DrawEntry>& dataEntry, DrawDataCandidates& dataCandidates, std::vector<StageDataHit>& stageData) {
    using Role = ShaderRecompiler::ProgramRole;
    if (useDrawEntries) {
        if (!registerKey) {
            drawKey = 0xcbf29ce484222325ull;
            const auto mix = [&](std::uint64_t value) {
                drawKey ^= value;
                drawKey *= 0x100000001b3ull;
            };
            mix(localDevice->Serial());
            mix(static_cast<std::uint64_t>(graphics.stages.path));
            mix(graphics.stages.registerValue);
            mix(graphics.stages.vertexWaveSize);
            mix(graphics.stages.fragmentWaveSize);
            mix(graphics.stages.mesh.has_value());
            if (graphics.stages.mesh) {
                const auto& mesh = *graphics.stages.mesh;
                for (const auto value : {mesh.inputPrimitive, mesh.primitivesPerGroup, mesh.verticesPerGroup, mesh.maxVertices, mesh.maxPrimitives, mesh.threadsPerGroup, mesh.ldsSizeDwords, mesh.provokingVertex, mesh.esgsItemSize, mesh.outputPrimitiveSize}) mix(value);
            }
            mix(graphics.stages.tessellation.has_value());
            if (graphics.stages.tessellation) {
                const auto& tess = *graphics.stages.tessellation;
                for (const auto value : {tess.inputControlPoints, tess.outputControlPoints, tess.domain, tess.partitioning, tess.outputTopology, tess.patchesPerGroup, tess.primitiveGroupSize, static_cast<std::uint32_t>(tess.breakAtInstance)}) mix(value);
                mix(tess.mergedHull);
            }
            mix(graphics.rectList);
            mix(graphics.rasterizerDiscard);
            mix(programs.size());
            for (std::size_t i = 0; i < programs.size(); ++i) {
                const auto& program = programs[i];
                mix(reinterpret_cast<std::uintptr_t>(program.snapshot.get()));
                mix(program.codeOffset);
                mix(static_cast<std::uint64_t>(roles[i]));
                mix(static_cast<std::uint64_t>(program.binary.stage));
                mix(program.userDataBase);
                mix(program.firstUserSgpr);
                mix(program.userData.size());
                for (const auto word : program.userData) mix(word);
                mix(vertexInfos[i].has_value());
                if (!vertexInfos[i]) continue;
                const auto& vertex = *vertexInfos[i];
                require(vertex.resourcesNum <= vertex.resources.size(), "vertex stage info resource count exceeds its table");
                mix(vertex.resourcesNum);
                mix(vertex.fetchAttribReg);
                mix(vertex.fetchBufferReg);
                mix(vertex.fetchEmbedded);
                mix(vertex.paClVsOutCntl);
                mix(vertex.pointList);
                mix(std::bit_cast<std::uint32_t>(vertex.pointSize));
                mix(std::bit_cast<std::uint32_t>(vertex.pointSizeMin));
                mix(std::bit_cast<std::uint32_t>(vertex.pointSizeMax));
                for (std::uint32_t r = 0; r < vertex.resourcesNum; ++r) {
                    for (const auto field : vertex.resources[r].fields) mix(field);
                    const auto& destination = vertex.resourcesDst[r];
                    mix(static_cast<std::uint32_t>(destination.registerStart));
                    mix(static_cast<std::uint32_t>(destination.registersNum));
                    mix(static_cast<std::uint32_t>(destination.attrId));
                    mix(destination.fetchIndex);
                }
            }
            require(pixel.interpolatorCount <= pixel.interpolatorSettings.size(), "pixel stage info interpolator count exceeds its table");
            mix(pixel.interpolatorCount);
            for (std::uint32_t i = 0; i < pixel.interpolatorCount; ++i) mix(pixel.interpolatorSettings[i]);
            mix(pixel.inputAddr);
            for (const bool flag : {pixel.wave32, pixel.hasPerspectiveCenterVgpr, pixel.perspectiveCentroid, pixel.posX, pixel.posY, pixel.posZ, pixel.posW, pixel.frontFace, pixel.frontFaceAllBits, pixel.ancillary, pixel.sampleShading, pixel.noPerspective, pixel.linearCentroid, pixel.pixelKillEnable, pixel.depthExportEnable, pixel.sampleMaskExportEnable, pixel.earlyZ, pixel.executeOnNoop}) mix(flag);
            mix(static_cast<std::uint64_t>(pixel.conservativeZExport));
            mix(pixel.orderedPixelShader);
            mix(pixel.stencilRefExportEnable);
            mix(pixel.stencilOperationValueExport);
            mix(std::bit_cast<std::uint32_t>(pixel.depthExportMin));
            mix(std::bit_cast<std::uint32_t>(pixel.depthExportMax));
            for (const auto value : pixel.targetOutputMode) mix(value);
            for (const auto value : pixel.targetExportMapping) mix(value);
            for (const auto value : pixel.targetExportPacking) mix(static_cast<std::uint64_t>(value));
            mix(pixel.dualSourceBlend);
            mix(pixel.reversedBlend);
            mix(pixel.targetMerges.size());
            for (const auto& merge : pixel.targetMerges) {
                mix(merge.target);
                mix(merge.mask);
                mix(merge.samples);
                mix(static_cast<std::uint64_t>(merge.packing));
                mix(merge.blendControl);
                mix(merge.layered);
                mix(merge.firstSlice);
                mix(merge.lastSlice);
                for (const auto word : merge.descriptor) mix(word);
                mix(merge.depthStencilWrites);
                mix(merge.depthStencilTests);
                if (!merge.blend.enabled) continue;
                const auto& blend = merge.blend;
                mix(blend.snorm);
                mix(blend.colorSource);
                mix(blend.colorDestination);
                mix(blend.colorOperation);
                mix(blend.alphaSource);
                mix(blend.alphaDestination);
                mix(blend.alphaOperation);
                mix(blend.writeMask);
                for (const auto word : blend.constants) mix(word);
            }
            mix(pixel.alphaToCoverage);
            mix(static_cast<std::uint64_t>(pixel.rasterPrimitive));
            mix(pixel.ditheredAlphaToCoverage);
            mix(pixel.ditheredAlphaSamples);
            mix(pixel.quadPixelMask);
            for (const auto mask : pixel.quadSampleMask) mix(mask);
            mix(pixel.iteratesSamples);
            mix(static_cast<std::uint64_t>(pixel.positionLocation));
            for (const auto location : pixel.sampleLocations) mix(location);
            for (const auto order : pixel.sampleOrder) mix(order);
            std::lock_guard cacheLock(drawCacheMutex);
            ++drawEntryCounters.lookups;
            const auto found = drawCache.find(drawKey);
            if (found != drawCache.end()) entry = found->second;
            else ++drawEntryCounters.absent;
            maybeReportDrawCache(profile);
        }
        if (entry != nullptr) {
            const auto waitedBeforeValidate = profile ? Graphics::Recorder::ThreadWaitedMs() : 0.0;
            std::optional<DrawMiss> miss;
            std::vector<std::size_t> ranks(programs.size(), 0);
            std::uint64_t stageValidations = 0, stageEqual = 0, compared = 0, imagesFlushed = 0, runsSynced = 0, dataStages = 0, dataWords = 0;
            std::vector<std::pair<std::uint32_t, std::uint32_t>> liveData;
            auto* live = drawStageDataHits() ? &liveData : nullptr;
            if (entry->stages.size() != programs.size()) miss = DrawMiss::Stages;

            std::uint32_t cursor = 0;
            {
                const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);

                std::optional<SampledReadScope> sampling;
                for (std::size_t i = 0; !miss && i < programs.size(); ++i) {
                    if (roles[i] == Role::GeometryBack) continue;
                    ++stageValidations;
                    cursor = Graphics::StagePushOffset(cursor, programs[i].binary.stage, localDevice->GraphicsPipelineLibraries());
                    const auto& variants = entry->stages[i];
                    auto outcome = EntryOutcome::Differing;
                    bool anyLayout = false;
                    for (std::size_t rank = 0; rank < variants.size(); ++rank) {
                        const auto& variant = variants[rank];
                        if (variant->registeredViews != nullptr && !CaptureRegisteredViewsCurrent(*variant->registeredViews)) continue;
                        if (variant->pushOffset != cursor) continue;
                        if (programs[i].binary.stage == ShaderRecompiler::ShaderStage::Mesh && !std::equal(variant->meshIndexDescriptor.begin(), variant->meshIndexDescriptor.end(), programs[i].userData.begin() + ShaderRecompiler::MeshIndexBufferUserWord)) {
                            if (!anyLayout) outcome = EntryOutcome::Differing;
                            anyLayout = true;
                            continue;
                        }
                        auto& regions = matchedRegions[i];
                        regions.clear();
                        appendEntryRegions(*variant, regions);
                        ++compared;
                        auto result = validateVariant(programs[i].binary.codeAddress, submission.queue, *variant, regions, imagesFlushed, runsSynced, sampling, live);
                        if (gateRetry() && (result == EntryOutcome::PublishMoved || result == EntryOutcome::PendingMoved)) result = validateVariant(programs[i].binary.codeAddress, submission.queue, *variant, regions, imagesFlushed, runsSynced, sampling, live);
                        if (!anyLayout) outcome = result;
                        anyLayout = true;
                        if (result != EntryOutcome::Equal && result != EntryOutcome::EqualData) continue;
                        if (result == EntryOutcome::EqualData) {
                            auto& hit = stageData[i];
                            hit.words = variant->words;
                            for (const auto& [position, value] : liveData) hit.words[position] = value;
                            regions.clear();
                            appendEntryRegions(*variant, regions, &hit.words);
                            hit.compiled = patchDataWords(*variant, hit.words);
                            ++dataStages;
                            dataWords += liveData.size();
                        }
                        matched[i] = variant;
                        ranks[i] = rank;
                        ++stageEqual;
                        cursor += static_cast<std::uint32_t>(variant->compiled->pushConstants.size());
                        break;
                    }
                    if (matched[i] != nullptr) continue;
                    if (!anyLayout) miss = DrawMiss::Layout;
                    else if (outcome == EntryOutcome::QueuedStores) miss = DrawMiss::QueuedStores;
                    else if (outcome != EntryOutcome::Differing) miss = DrawMiss::Gate;
                    else miss = roles[i] == Role::Fragment ? DrawMiss::FragmentDiffering : i == 0 ? DrawMiss::FrontDiffering : DrawMiss::OtherDiffering;
                }
            }
            drawHit = !miss;
            std::lock_guard cacheLock(drawCacheMutex);
            auto& counters = drawEntryCounters;
            counters.stageValidations += stageValidations;
            counters.stageEqual += stageEqual;
            counters.variantsCompared += compared;
            if (drawHit) {
                ++counters.hits;
                counters.stageDataHits += dataStages;
                counters.stageDataWords += dataWords;
                if (registerKey) ++counters.registerKeyHits;
                ++drawCacheHits;
                bool rotate = false;
                for (std::size_t i = 0; i < programs.size(); ++i) {
                    if (matched[i] == nullptr) continue;
                    ++counters.variantHitsByRank[ranks[i]];
                    if (ranks[i] != 0) rotate = true;
                }

                const auto again = drawCache.find(drawKey);
                if (again != drawCache.end() && again->second == entry) {
                    if (rotate) {
                        auto rotated = std::make_shared<DrawEntry>();
                        rotated->decode = entry->decode;
                        rotated->stages = entry->stages;
                        rotated->shape = entry->shape;
                        rotated->recipes.store(entry->recipes.load());
                        for (std::size_t i = 0; i < programs.size(); ++i) {
                            if (ranks[i] == 0) continue;
                            auto& variants = rotated->stages[i];
                            variants.erase(variants.begin() + static_cast<std::ptrdiff_t>(ranks[i]));
                            variants.insert(variants.begin(), matched[i]);
                        }
                        rotated->touched = entry->touched;
                        rotated->order = entry->order;
                        again->second = std::move(rotated);
                    }
                    if (drawCacheHits - again->second->touched > drawCacheEntries() / 8) {
                        drawOrder.splice(drawOrder.begin(), drawOrder, again->second->order);
                        again->second->touched = drawCacheHits;
                        ++counters.touches;
                    }
                }
                if (verifyDrawEntries()) {
                    ++counters.verifyHits;
                    verifyHit = true;
                    drawHit = false;
                }
                if (dataStages != 0 && verifyDrawDataHits()) drawHit = false;
            } else {
                ++counters.misses[static_cast<std::size_t>(*miss)];
                if (registerKey && entry->decode != nullptr) ++counters.decodePartial;
            }
            if (profile) {

                phaseTiming.Phase(DrawRowKeyLookupValidate);
                const auto waited = std::min(Graphics::Recorder::ThreadWaitedMs() - waitedBeforeValidate, phaseMs[DrawRowKeyLookupValidate]);
                phaseMs[DrawRowKeyLookupValidate] -= waited;
                phaseMs[DrawRowValidateWait] += waited;
                counters.validateUs += phaseMs[DrawRowKeyLookupValidate] * 1000;
            }
        } else if (dataEntry != nullptr && graphics.stages.path == Graphics::ShaderPath::Vertex && dataEntry->stages.size() == programs.size() && dataEntry->decode->programs.size() == programs.size()) {
            std::uint64_t imagesFlushed = 0, runsSynced = 0;
            std::vector<std::pair<std::uint32_t, std::uint32_t>> forwarded;
            DataReuseCounts reuse;
            const struct ReuseNote {
                DataReuseCounts& counts;
                ~ReuseNote() { NoteDataReuse(counts); }
            } reuseNote{reuse};
            const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
            std::optional<SampledReadScope> sampling;
            for (std::size_t i = 0; i < programs.size(); ++i) {
                if (roles[i] == Role::GeometryBack || programs[i].userData.size() != dataEntry->decode->programs[i].userData.size()) continue;
                const auto& now = programs[i].userData;
                const auto& was = dataEntry->decode->programs[i].userData;
                auto& candidates = dataCandidates[i];
                for (const auto& variant : dataEntry->stages[i]) {
                    if (variant->registeredViews != nullptr && !CaptureRegisteredViewsCurrent(*variant->registeredViews)) continue;
                    bool usedEqual = true;
                    forwarded.clear();
                    for (std::size_t w = 0; w < now.size() && usedEqual; ++w) {
                        if (now[w] == was[w] || (w < 64u && ((variant->userDataMask >> w) & 1u) == 0)) continue;
                        const auto slot = forwardedSlot(*variant, w);
                        if (slot < 0) usedEqual = false;
                        else forwarded.emplace_back(static_cast<std::uint32_t>(slot), now[w]);
                    }
                    if (!usedEqual) {
                        ++reuse.userWords;
                        continue;
                    }
                    if ((i == 0 && variant->pushOffset != 0) || std::any_of(candidates.begin(), candidates.end(), [&](const auto& candidate) { return candidate.first->pushOffset == variant->pushOffset; })) continue;
                    std::vector<ShaderRecompiler::MemoryRegion> regions;
                    appendEntryRegions(*variant, regions);
                    auto result = validateVariant(programs[i].binary.codeAddress, submission.queue, *variant, regions, imagesFlushed, runsSynced, sampling);
                    if (gateRetry() && (result == EntryOutcome::PublishMoved || result == EntryOutcome::PendingMoved)) result = validateVariant(programs[i].binary.codeAddress, submission.queue, *variant, regions, imagesFlushed, runsSynced, sampling);
                    if (result != EntryOutcome::Equal) {
                        ++reuse.memory;
                        continue;
                    }
                    if (forwarded.empty()) {
                        ++reuse.plain;
                        candidates.emplace_back(variant, std::move(regions));
                        continue;
                    }
                    ++reuse.patched;
                    candidates.emplace_back(forwardedCopy(*variant, forwarded), std::move(regions));
                }
            }
        }
        phaseTiming.Phase(DrawRowKeyLookupValidate);
    }
}

}
