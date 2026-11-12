#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <span>
#include <sstream>
#include <stdexcept>
#include <exception>
#include <typeinfo>
#include <array>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <string>

namespace AgcDriver::DriverDetail {

namespace {

void TraceMaterializationFailure(std::uint64_t token, const Submission& submission, std::size_t packetOffset, std::size_t stage, const char* phase, const std::exception_ptr& failure) {
    if (token == 0) return;
    const char* type = "unknown";
    std::string reason = "unknown exception";
    try {
        if (failure != nullptr) std::rethrow_exception(failure);
    } catch (const std::exception& error) {
        type = typeid(error).name();
        reason = error.what();
    } catch (...) {
    }
    try {
        AgcDriver::CaptureTrace::Log("draw-materialization-failed token=%llu submission=%llu queue=%x offset=%zu stage=%zu phase=%s exception=%s reason=%.256s",
                static_cast<unsigned long long>(token), static_cast<unsigned long long>(submission.serial), submission.queue, packetOffset, stage, phase, type, reason.c_str());
    } catch (...) {
    }
}

void TraceShaderRequestFailure(std::uint64_t token, const Submission& submission, std::size_t packetOffset, const DrawProgram& program, const std::exception_ptr& failure) noexcept {
    if (token == 0) return;
    try {
        std::string reason = "unknown exception";
        const char* type = "unknown";
        try {
            if (failure != nullptr) std::rethrow_exception(failure);
        } catch (const std::exception& error) {
            type = typeid(error).name();
            reason = error.what();
        } catch (...) {
        }
        AgcDriver::CaptureTrace::LogBestEffort("draw-shader-request-failed token=%llu submission=%llu queue=%x offset=%zu snapshot=0x%llx code=%llx codeOffset=%zu stage=%u type=%u reason=%.256s",
                static_cast<unsigned long long>(token), static_cast<unsigned long long>(submission.serial), submission.queue, packetOffset,
                static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(program.snapshot.get())),
                static_cast<unsigned long long>(program.snapshot->codeAddress), program.codeOffset,
                static_cast<unsigned>(program.binary.stage), static_cast<unsigned>(program.snapshot->type), reason.c_str());
    } catch (...) {
    }
}

}

std::shared_ptr<const ShaderRecompiler::RecompileResult> Driver::materializeDrawStage(std::size_t i, std::uint32_t pushOffset, const QueueState& queue, const Submission& submission, std::size_t packetOffset, std::uint64_t traceToken, const std::vector<DrawProgram>& programs, const Graphics::State& graphics, const ShaderRecompiler::ShaderPixelStageInfo& pixel, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, std::vector<ShaderRecompiler::MemoryRegion>& memory, const std::vector<ShaderRecompiler::LinkedProgram>& linked, const Pm4::DrawParameters& drawParameters, const std::shared_ptr<VulkanDevice>& localDevice, ShaderMemory& shaderMemory, std::vector<StageCapture>& stageCaptures, std::vector<bool>& recompiled, bool drawHit, const std::vector<std::shared_ptr<DispatchVariant>>& matched, const std::vector<std::vector<ShaderRecompiler::MemoryRegion>>& matchedRegions, bool profile, std::uint64_t dumpTarget, std::uint64_t dumpSlot1, std::uint64_t& captures, DrawPhaseTiming& phaseTiming, std::array<double, DrawDriverPhaseCount>& phaseMs, std::string& rejected, const std::optional<ShaderRecompiler::MeshBarycentricLayout>& meshBarycentrics) try {
    PerformanceTimer timing("Shader.DrawStage");
    using Stage = ShaderRecompiler::ShaderStage;
    phaseTiming.Phase(DrawRowVectors);
    const auto& program = programs[i];
    const auto waveSize = Graphics::StageWaveSize(graphics.stages, program.binary.stage);
    ShaderRecompiler::RecompileRequest request{
        program.binary,
        {waveSize, program.firstUserSgpr, program.userData, std::nullopt, program.binary.stage == Stage::Fragment ? std::optional(pixel) : std::nullopt, vertexInfos[i], memory, RegisteredFloatMode(*program.snapshot)},
        localDevice->Target(),
        {0, 0, pushOffset, Graphics::StagePushLimit(graphics.stages, pushOffset, graphics.depthRemap.mode != 0u) - pushOffset % Graphics::PipelinePushSlotBytes},
        ShaderRecompiler::GraphicsCompileContext{program.firstUserSgpr, linked, Graphics::StageMeshConfiguration(graphics.stages, program.binary.stage), graphics.stages.tessellation, {drawParameters.indexAddress, drawParameters.indexCount, drawParameters.indexSize, drawParameters.instanceCount}, program.binary.stage == Stage::Mesh ? meshBarycentrics : std::nullopt}
    };
    if (graphics.depthRemap.mode != 0u && pushOffset < Graphics::PipelinePushSlotBytes) {
        request.layout.pushConstantSizeBytes = std::min(request.layout.pushConstantSizeBytes, ShaderRecompiler::DepthRemapPushOffsetBytes - pushOffset);
    }
    const auto waitedBefore = traceCapSync() || profile ? Graphics::Recorder::ThreadWaitedMs() : 0.0;
    auto invocation = [&] {
        try {
            return InvocationFor(*program.snapshot, program.codeOffset, request);
        } catch (...) {
            TraceShaderRequestFailure(traceToken, submission, packetOffset, program, std::current_exception());
            throw;
        }
    }();
    timing.Mark("prepared_invocation");
    auto& stageCapture = stageCaptures[i];
    stageCapture.forgetSerial = GuestMemory::ForgetSerial();
    stageCapture.pushOffset = pushOffset;
    const auto named = [&](const std::exception& error, std::span<const ShaderRecompiler::MemoryRegion> regions) {
        static const bool dumpShaders = std::getenv("APS5_DUMP_SHADERS") != nullptr;
        std::string dump;
        if (dumpShaders) {
            try {
                auto failed = request;
                failed.context.memory = regions;
                dump = dumpRequest(program.binary.codeAddress, failed, "_failed");
            } catch (...) {}
        }
        std::ostringstream where;
        where << "shader 0x" << std::hex << program.binary.codeAddress << " stage " << std::dec << static_cast<unsigned>(program.binary.stage);
        if (!dump.empty()) where << " (" << dump << ")";
        where << ": ";
        return std::runtime_error(where.str() + error.what());
    };
    const auto capture = [&] {
        const SampledReadScope sampling(evidenceReads);
        ShaderMemory::SetTraceProgram(program.binary.codeAddress);
        try {
            return CaptureInvocation(shaderMemory, *program.snapshot, program.codeOffset, request, invocation);
        } catch (const std::exception& error) {
            throw named(error, shaderMemory.Regions());
        }
    }();

    stageCapture.regions = shaderMemory.TakeRecentRegions();
    stageCapture.supplied = shaderMemory.SuppliedBlocks();
    stageCapture.userDataUse = capture->userDataUse;
    stageCapture.registeredViews = std::make_shared<const std::vector<ShaderRecompiler::RegisteredImageView>>(capture->snapshot.registeredImages);
    recompiled[i] = true;
    memory = shaderMemory.Regions();

    for (std::size_t j = 0; j < programs.size(); ++j) {
        if (matched[j] != nullptr && !recompiled[j] && (drawHit || j < i)) memory.insert(memory.end(), matchedRegions[j].begin(), matchedRegions[j].end());
    }
    request.context.memory = memory;
    static const bool dumpGraphics = std::getenv("APS5_DUMP_GRAPHICS_SHADERS") != nullptr;
    if (dumpGraphics) static_cast<void>(dumpRequest(program.binary.codeAddress, request));
    if (traceCapSync()) traceCapture("draw-capture", program.binary.codeAddress, submission.queue, memory, Graphics::Recorder::ThreadWaitedMs() - waitedBefore);
    if (profile) {
        ++captures;
        phaseTiming.Phase(DrawRowCapture);

            const auto waited = std::min(Graphics::Recorder::ThreadWaitedMs() - waitedBefore, phaseMs[DrawRowCapture]);
            phaseMs[DrawRowCapture] -= waited;
            phaseMs[DrawRowCaptureHookWaits] += waited;
        }
        if (dumpTarget != 0) {

        const auto slot0 = (static_cast<std::uint64_t>(readRegister(queue.context, 0x390)) << 40u) | (static_cast<std::uint64_t>(readRegister(queue.context, 0x318)) << 8u);
        if ((graphics.hasColorTarget && graphics.color.address == dumpTarget) || slot0 == dumpTarget || program.binary.codeAddress == dumpTarget) static_cast<void>(dumpRequest(program.binary.codeAddress, request));
        }
        if (dumpSlot1 != 0) {
        const auto value = [&](std::uint32_t offset) -> std::uint64_t { const auto it = queue.context.find(offset); return it == queue.context.end() ? 0u : it->second; };
        const auto slot1 = (value(0x391) << 40u) | (value(0x327) << 8u);
        if (slot1 == dumpSlot1) {
            static_cast<void>(dumpRequest(program.binary.codeAddress, request));
            if (std::FILE* file = std::fopen("draw_slot1.regs", "w")) {
                for (const auto& [offset, value] : queue.context) std::fprintf(file, "context %x %08x\n", offset, value);
                for (const auto& [offset, value] : queue.userConfig) std::fprintf(file, "uconfig %x %08x\n", offset, value);
                for (const auto& [offset, value] : queue.shader) std::fprintf(file, "shader %x %08x\n", offset, value);
                std::fclose(file);
            }
        }
        }

        phaseTiming.Phase(DrawRowCapture);

    timing.Mark("capture_resources");
    try {
        stageCapture.compiled = invocation.Materialize(*capture);
        stageCapture.capture = capture;
    } catch (const std::exception& error) {
        TraceMaterializationFailure(traceToken, submission, packetOffset, i, "capture-or-materialize", std::current_exception());
        throw named(error, request.context.memory);
    }
    timing.Mark("materialize");
    const auto walked = capture->userDataUse.asValue | capture->userDataUse.asBase;
    auto mask = walked == 0 ? ~std::uint64_t{0} : walked | stageCapture.compiled->userDataRegisterReads;
    if (vertexInfos[i]) {
        for (const auto reg : {static_cast<std::uint32_t>(vertexInfos[i]->fetchAttribReg), static_cast<std::uint32_t>(vertexInfos[i]->fetchBufferReg)}) {
            mask |= reg + 1u < 64u ? std::uint64_t{3} << reg : ~std::uint64_t{0};
        }
    }
    stageCapture.userDataMask = mask;
    phaseTiming.Phase(DrawRowRecompile);
    return stageCapture.compiled;
} catch (const std::exception& error) {
    if (!SkipRefusedDraws()) throw;
    throw RefusedDraw(programs[i].binary.codeAddress, error.what());
}

void Driver::recordStageDataPositions(DispatchVariant& variant, const ShaderRecompiler::ResourceCapture& capture, const std::vector<Graphics::DecodeRead>& decodeReads) {
    const auto& compiled = *variant.compiled;
    for (std::size_t b = 0; b < compiled.bindings.size(); ++b) {
        if (compiled.bindings[b].role != ShaderRecompiler::DescriptorRole::FlattenedSrt) continue;
        variant.flatBinding = static_cast<std::uint32_t>(b);
        static_cast<void>(DataWordPositions(variant.runs, capture.readTrace.leaves, capture.readTrace.otherReads, variant.words, compiled.bindings[b].guestDescriptor, variant.dataPositions, variant.dataSlots));
        break;
    }
    if (variant.dataPositions.empty() || decodeReads.empty()) return;
    std::vector<bool> decoded(variant.words.size(), false);
    std::size_t offset = 0;
    for (const auto& [begin, end] : variant.runs) {
        for (const auto& read : decodeReads) {
            const auto from = std::max(begin, read.address);
            const auto to = std::min(end, read.address + read.size);
            if (from >= to) continue;
            for (auto word = (from - begin) / sizeof(std::uint32_t); word < (to - begin + sizeof(std::uint32_t) - 1) / sizeof(std::uint32_t); ++word) decoded[offset + static_cast<std::size_t>(word)] = true;
        }
        offset += static_cast<std::size_t>((end - begin) / sizeof(std::uint32_t));
    }
    std::size_t kept = 0;
    for (std::size_t k = 0; k < variant.dataPositions.size(); ++k) {
        if (decoded[variant.dataPositions[k]]) continue;
        variant.dataPositions[kept] = variant.dataPositions[k];
        variant.dataSlots[kept] = variant.dataSlots[k];
        ++kept;
    }
    variant.dataPositions.resize(kept);
    variant.dataSlots.resize(kept);
}

void Driver::cacheDrawStages(bool useDrawEntries, bool drawHit, const Pm4::DrawParameters& drawParameters, const std::optional<Graphics::IndirectDrawPath>& indirectCpu, const std::vector<DrawProgram>& programs, const std::vector<StageCapture>& stageCaptures, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, const std::vector<std::vector<Graphics::DecodeRead>>& decodeReads, bool verifyHit, const std::vector<std::shared_ptr<DispatchVariant>>& matched, const std::vector<StageDataHit>& stageData, std::vector<std::shared_ptr<DispatchVariant>>& fresh, std::uint64_t drawKey, bool registerKey, const std::shared_ptr<const DrawDecode>& decode, DrawPhaseTiming& phaseTiming, const std::vector<bool>& reused, const std::shared_ptr<DrawEntry>& entry, std::uint64_t shapeKey, bool shapeNeedsEntry) {
    const bool cacheable = useDrawEntries && !drawHit && (drawCacheIndirect() || !(drawParameters.indirect && indirectCpu));
    const bool admitted = cacheable && (verifyHit || admitDrawKey(drawKey, registerKey && drawDataHits() ? shapeKey : 0));
    const bool replaceShape = cacheable && !admitted && shapeNeedsEntry;
    if (admitted || replaceShape) {
        phaseTiming.Phase(DrawRowVectors);
        std::uint64_t unstable = 0, mismatches = 0, differingSameRuns = 0, differingWords = 0, differingRunsChanged = 0;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            const auto& stageCapture = stageCaptures[i];
            if (reused[i]) {
                fresh[i] = matched[i];
                continue;
            }
            if (stageCapture.compiled == nullptr || !CacheableResult(*stageCapture.compiled)) continue;
            auto variant = std::make_shared<DispatchVariant>();
            variant->compiled = stageCapture.compiled;
            variant->registeredViews = stageCapture.registeredViews;
            if (programs[i].binary.stage == ShaderRecompiler::ShaderStage::Mesh) {
                std::copy_n(programs[i].userData.begin() + ShaderRecompiler::MeshIndexBufferUserWord, variant->meshIndexDescriptor.size(), variant->meshIndexDescriptor.begin());
            }
            variant->shader = programs[i].snapshot;
            variant->forgetSerial = stageCapture.forgetSerial;
            variant->pushOffset = stageCapture.pushOffset;
            variant->userDataMask = stageCapture.userDataMask;
            if (vertexInfos[i]) variant->vertexInfo = std::make_shared<const ShaderRecompiler::ShaderVertexStageInfo>(*vertexInfos[i]);

            std::vector<ShaderRecompiler::MemoryRegion> regions(stageCapture.regions.begin(), stageCapture.regions.end());
            for (const auto& read : decodeReads[i]) regions.push_back({read.address, read.Bytes()});
            std::stable_sort(regions.begin(), regions.end(), [](const ShaderRecompiler::MemoryRegion& a, const ShaderRecompiler::MemoryRegion& b) { return a.guestAddress < b.guestAddress; });
            const bool relocatable = decodeReads[i].empty() && (!vertexInfos[i] || vertexInfos[i]->resourcesNum == 0);
            tagVariantRuns(*variant, regions, stageCapture.userDataUse, relocatable, programs[i].hiddenUserWords, stageCapture.supplied);
            if (relocatable) tagUserValueWords(*variant, programs[i]);
            if (drawStageDataHits() && stageCapture.capture != nullptr && !stageCapture.capture->readTrace.leaves.empty()) recordStageDataPositions(*variant, *stageCapture.capture, decodeReads[i]);
            if (traceDrawCache() && entry != nullptr && matched[i] == nullptr && i < entry->stages.size()) {
                const auto& variants = entry->stages[i];
                const auto front = std::find_if(variants.begin(), variants.end(), [&](const std::shared_ptr<DispatchVariant>& kept) { return kept->pushOffset == variant->pushOffset; });
                if (front != variants.end() && (*front)->runs == variant->runs && (*front)->words.size() == variant->words.size()) {
                    ++differingSameRuns;
                    for (std::size_t w = 0; w < variant->words.size(); ++w) differingWords += (*front)->words[w] != variant->words[w] ? 1u : 0u;
                } else if (front != variants.end()) {
                    ++differingRunsChanged;
                }
            }
            const auto sameContent = [](const DispatchVariant& a, const std::vector<std::uint32_t>& words, const DispatchVariant& b) {
                const auto flatten = [](const DispatchVariant& source, const std::vector<std::uint32_t>& values) {
                    std::vector<std::pair<std::uint64_t, std::uint32_t>> flat;
                    std::size_t word = 0;
                    for (const auto& [begin, end] : source.runs) {
                        for (auto address = begin; address < end && word < values.size(); address += sizeof(std::uint32_t)) flat.emplace_back(address, values[word++]);
                    }
                    std::sort(flat.begin(), flat.end());
                    return flat;
                };
                return (a.runs == b.runs && words == b.words) || flatten(a, words) == flatten(b, b.words);
            };
            const auto sameForwardedWords = [](const ShaderRecompiler::RecompileResult& a, const ShaderRecompiler::RecompileResult& b, std::uint32_t hidden) {
                if (a.pushConstants.size() != b.pushConstants.size() || a.shaderDataUserWords != b.shaderDataUserWords) return false;
                for (std::size_t slot = 0; slot < a.shaderDataUserWords.size() && (slot + 1) * sizeof(std::uint32_t) <= a.pushConstants.size(); ++slot) {
                    if (a.shaderDataUserWords[slot] >= hidden && std::memcmp(a.pushConstants.data() + slot * sizeof(std::uint32_t), b.pushConstants.data() + slot * sizeof(std::uint32_t), sizeof(std::uint32_t)) != 0) return false;
                }
                return true;
            };
            if (verifyHit && matched[i] != nullptr) {
                const auto& words = stageData[i].compiled != nullptr ? stageData[i].words : matched[i]->words;
                const auto& compiled = stageData[i].compiled != nullptr ? *stageData[i].compiled : *matched[i]->compiled;
                if (!sameContent(*matched[i], words, *variant) || !sameForwardedWords(compiled, *variant->compiled, programs[i].hiddenUserWords)) {
                    ++mismatches;
                    static std::atomic<std::uint64_t> reports{0};
                    if (reports.fetch_add(1) < 20) std::fprintf(stderr, "[draw-cache] verify: stage %zu (program 0x%llx) of a hit captured differently: %zu runs / %zu words matched, %zu / %zu fresh\n", i, static_cast<unsigned long long>(programs[i].binary.codeAddress), matched[i]->runs.size(), words.size(), variant->runs.size(), variant->words.size());
                }
            }
            if (insertCompare()) {
                const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
                if (!captureStable(stageCapture.regions)) {
                    ++unstable;
                    continue;
                }
            }
            fresh[i] = std::move(variant);
        }
        insertDrawEntry(drawKey, fresh, registerKey ? decode : nullptr, registerKey && drawDataHits() ? shapeKey : 0, replaceShape);
        std::uint64_t dataInserts = 0, dataPositions = 0;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (fresh[i] == nullptr || reused[i] || fresh[i]->dataPositions.empty()) continue;
            ++dataInserts;
            dataPositions += fresh[i]->dataPositions.size();
        }
        if (unstable != 0 || mismatches != 0 || differingSameRuns != 0 || differingRunsChanged != 0 || dataInserts != 0) {
            std::lock_guard cacheLock(drawCacheMutex);
            drawEntryCounters.stageDataInserts += dataInserts;
            drawEntryCounters.stageDataPositions += dataPositions;
            drawEntryCounters.unstable += unstable;
            drawEntryCounters.verifyMismatches += mismatches;
            drawEntryCounters.differingSameRuns += differingSameRuns;
            drawEntryCounters.differingWords += differingWords;
            drawEntryCounters.differingRunsChanged += differingRunsChanged;
        }
        phaseTiming.Phase(DrawRowKeyLookupValidate);
    }
}

// Debug aid: APS5_TRACE_DRAW_REUSE=1 compares every freshly captured draw stage with the previous
// capture of the same program (code, offset, push offset) and prints, every 10 s, what differed:
// the user words, the read addresses, the snapshot's descriptors (address fields or other fields)
// and flat words, and the result's variant, specialization, binding shapes and push data. It
// measures which captures a cache keyed on the program alone could serve by patching data.
void NoteDrawReuse(const std::vector<DrawProgram>& programs, const std::vector<StageCapture>& stageCaptures, const std::vector<bool>& recompiled) {
    static const bool enabled = std::getenv("APS5_TRACE_DRAW_REUSE") != nullptr;
    if (!enabled) return;
    struct Last {
        std::vector<std::uint32_t> userData;
        std::vector<std::pair<std::uint64_t, std::size_t>> regions;
        ShaderRecompiler::ResourceSnapshot snapshot;
        std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
    };
    enum Count : std::size_t { Stages, First, Identical, UserDiffer, UserBaseDiffer, RegionsSame, RegionsShifted, RegionsOther, BuffersAddr, BuffersOther, ImagesAddr, ImagesOther, SamplersDiffer, FlatDiffer, SnapshotSame, Variant, Specialization, BindingShape, GuestDescriptor, PushData, DataOnly, DataOnlyRegionsSame, Total };
    static std::mutex mutex;
    static std::unordered_map<std::uint64_t, Last> last;
    static std::array<std::uint64_t, Total> counts{};
    static auto reported = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex);
    for (std::size_t i = 0; i < programs.size() && i < stageCaptures.size(); ++i) {
        const auto& capture = stageCaptures[i];
        if (i >= recompiled.size() || !recompiled[i] || capture.compiled == nullptr || capture.capture == nullptr) continue;
        ++counts[Stages];
        const auto key = (static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(programs[i].snapshot.get())) * 0x9e3779b97f4a7c15ull) ^ (static_cast<std::uint64_t>(programs[i].codeOffset) << 20u) ^ capture.pushOffset;
        Last now;
        now.userData = programs[i].userData;
        for (const auto& region : capture.regions) now.regions.emplace_back(region.guestAddress, region.bytes.size());
        std::sort(now.regions.begin(), now.regions.end());
        now.snapshot = capture.capture->snapshot;
        now.compiled = capture.compiled;
        auto [it, inserted] = last.try_emplace(key);
        if (inserted) {
            ++counts[First];
            it->second = std::move(now);
            continue;
        }
        const auto& was = it->second;
        const bool userSame = was.userData == now.userData;
        if (!userSame) {
            ++counts[UserDiffer];
            const auto base = capture.userDataUse.asBase;
            for (std::size_t w = 0; w < now.userData.size() && w < was.userData.size() && w < 64; ++w) {
                if (now.userData[w] != was.userData[w] && ((base >> w) & 1u) != 0) {
                    ++counts[UserBaseDiffer];
                    break;
                }
            }
        }
        const bool regionsSame = was.regions == now.regions;
        if (regionsSame) {
            ++counts[RegionsSame];
        } else {
            bool shifted = was.regions.size() == now.regions.size();
            for (std::size_t r = 0; shifted && r < now.regions.size(); ++r) shifted = was.regions[r].second == now.regions[r].second;
            ++counts[shifted ? RegionsShifted : RegionsOther];
        }
        const auto descriptors = [&](const std::vector<ShaderRecompiler::DescriptorValue>& a, const std::vector<ShaderRecompiler::DescriptorValue>& b, bool image, Count addr, Count other) {
            if (a.size() != b.size()) {
                ++counts[other];
                return;
            }
            bool address = false, rest = false;
            for (std::size_t d = 0; d < a.size(); ++d) {
                if (a[d] == b[d]) continue;
                for (std::size_t w = 0; w < 8; ++w) {
                    const auto diff = a[d].dwords[w] ^ b[d].dwords[w];
                    if (diff == 0) continue;
                    const std::uint32_t addressMask = w == 0 ? ~0u : w == 1 ? (image ? 0xffu : 0xffffu) : 0u;
                    if ((diff & addressMask) != 0) address = true;
                    if ((diff & ~addressMask) != 0) rest = true;
                }
                if (a[d].dwordCount != b[d].dwordCount || a[d].availableDwordMask != b[d].availableDwordMask) rest = true;
            }
            if (address) ++counts[addr];
            if (rest) ++counts[other];
        };
        descriptors(was.snapshot.buffers, now.snapshot.buffers, false, BuffersAddr, BuffersOther);
        descriptors(was.snapshot.images, now.snapshot.images, true, ImagesAddr, ImagesOther);
        if (was.snapshot.samplers != now.snapshot.samplers) ++counts[SamplersDiffer];
        if (was.snapshot.flattenedSrt != now.snapshot.flattenedSrt) ++counts[FlatDiffer];
        const bool snapshotSame = was.snapshot.buffers == now.snapshot.buffers && was.snapshot.images == now.snapshot.images && was.snapshot.samplers == now.snapshot.samplers && was.snapshot.flattenedSrt == now.snapshot.flattenedSrt;
        if (snapshotSame) ++counts[SnapshotSame];
        const auto& a = *was.compiled;
        const auto& b = *now.compiled;
        const bool variant = a.variantId != b.variantId;
        const bool specialization = a.specializationId != b.specializationId || a.specialization.size() != b.specialization.size();
        bool shape = a.bindings.size() != b.bindings.size();
        bool guest = false;
        for (std::size_t k = 0; !shape && k < a.bindings.size(); ++k) {
            const auto& x = a.bindings[k];
            const auto& y = b.bindings[k];
            shape = x.kind != y.kind || x.role != y.role || x.descriptorSet != y.descriptorSet || x.binding != y.binding || x.count != y.count || x.imageShapes.size() != y.imageShapes.size() || x.imageShape.has_value() != y.imageShape.has_value() || x.bufferWritten != y.bufferWritten || x.imageWritten != y.imageWritten;
            if (x.guestDescriptor != y.guestDescriptor) guest = true;
        }
        const bool push = a.pushConstants != b.pushConstants;
        if (variant) ++counts[Variant];
        if (specialization) ++counts[Specialization];
        if (shape) ++counts[BindingShape];
        if (guest) ++counts[GuestDescriptor];
        if (push) ++counts[PushData];
        if (!variant && !specialization && !shape) {
            ++counts[DataOnly];
            if (regionsSame) ++counts[DataOnlyRegionsSame];
        }
        if (userSame && regionsSame && snapshotSame && !variant && !specialization && !shape && !guest && !push) ++counts[Identical];
        it->second = std::move(now);
    }
    if (last.size() > 65536) last.clear();
    const auto clock = std::chrono::steady_clock::now();
    if (clock - reported < std::chrono::seconds(10)) return;
    reported = clock;
    const auto c = [&](Count index) { return static_cast<unsigned long long>(counts[index]); };
    std::fprintf(stderr, "[draw-reuse] (10 s) %llu fresh stage captures (%llu first of their program, %llu identical to the previous one); vs the previous capture of the program: user words differ %llu (a walked base among them %llu); read regions same %llu / same sizes shifted %llu / other %llu; snapshot same %llu; buffers differ in address %llu / other fields %llu; images address %llu / other %llu; samplers %llu; flat words %llu; result: variant %llu, specialization %llu, binding shape %llu, guest descriptors %llu, push data %llu; data-only (same variant, specialization, binding shapes) %llu (with the same read regions %llu)\n",
        c(Stages), c(First), c(Identical), c(UserDiffer), c(UserBaseDiffer), c(RegionsSame), c(RegionsShifted), c(RegionsOther), c(SnapshotSame), c(BuffersAddr), c(BuffersOther), c(ImagesAddr), c(ImagesOther), c(SamplersDiffer), c(FlatDiffer), c(Variant), c(Specialization), c(BindingShape), c(GuestDescriptor), c(PushData), c(DataOnly), c(DataOnlyRegionsSame));
    counts = {};
}

}
