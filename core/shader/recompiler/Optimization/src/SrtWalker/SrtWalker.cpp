#include "Optimization/SrtWalker.hpp"
#include "Optimization/SrtWalker/SrtDescriptorEvaluation.hpp"
#include "Optimization/SrtWalker/SrtEvaluator.hpp"
#include "Optimization/SrtWalker/SrtPlanBuilder.hpp"
#include "Optimization/SrtWalker/SrtRuntimeValidator.hpp"
#include "prx/libc/include/HostThreadSlot.hpp"
#include "Recompiler.hpp"

#include <atomic>
#include <cstdio>
#include <stdexcept>

namespace ShaderRecompiler {

struct SrtEvaluatingBranchSlot {};

bool SrtEvaluatingBranch() {
    return HostThreadSlot<bool, SrtEvaluatingBranchSlot>::Get();
}

void SetSrtEvaluatingBranch(bool evaluating) {
    HostThreadSlot<bool, SrtEvaluatingBranchSlot>::Set(evaluating);
}

void SrtWalker::BuildPlan(IrProgram& program) const {
    if (program.Resources().resourceTrackingComplete) {
        throw std::runtime_error("shader SRT planning failed: cannot rebuild SRT after resource tracking");
    }
    program.Resources().srtPlanComplete = false;
    program.Resources().gpuSrt = GpuSrtEligible(program);
    if (program.Resources().gpuSrt) {
        static std::atomic<std::uint32_t> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 400) std::fprintf(stderr, "[gpu-srt] shader %016llx stage %u: SRT left to the GPU\n", static_cast<unsigned long long>(program.Resources().shaderHash), static_cast<unsigned>(program.Resources().stage));
        program.Resources().srtReads.clear();
        program.Metadata().dynamicReads.clear();
    } else {
        Detail::PlanBuilder(program).Run();
    }
    program.Resources().srtPlanComplete = true;
}

bool SrtWalker::GpuSrtEligible(const IrProgram& program) {
    const auto stages = GpuSrtStages();
    if (stages == 0u || (stages == 1u && program.Resources().stage != IrShaderStage::Compute)) return false;
    for (const auto& block : program.Blocks()) {
        for (const IrValue* inst : block->Instructions()) {
            const auto op = inst->Opcode();
            if (op == IrOpcode::GetImageResource || op == IrOpcode::GetSamplerResource || op == IrOpcode::ImageBvhIntersectRay || ImageOpcodeInfoOf(op).access != ImageAccess::None) return false;
        }
    }
    return true;
}

bool SrtWalker::ValidateRuntimeValue(const IrResourcePlan& program, const IrValue* value, RuntimeValueType type) const {
    return Detail::RuntimeValidator(program, type).Run(const_cast<IrValue*>(value));
}

void SrtWalker::EvaluateUniformValues(const IrResourcePlan& program, std::span<IrValue* const> values, const SrtRuntime& runtime, std::span<std::uint32_t> results) const {
    if (values.size() != results.size()) {
        throw std::runtime_error("SrtWalker::EvaluateUniformValues value and result counts differ");
    }
    auto clean = runtime;
    clean.readMemory = runtime.readSpecializationMemory != nullptr ? runtime.readSpecializationMemory : +[](void*, std::uint64_t, std::uint32_t*) { return false; };
    Detail::Evaluator evaluator(program, clean);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!evaluator.Evaluate(Detail::IrNode(values[i]), results[i])) {
            throw std::runtime_error("SrtWalker::EvaluateUniformValues failed to evaluate a uniform value");
        }
    }
}

void SrtWalker::EvaluateDescriptorSource(const IrResourcePlan& program, std::uint32_t source, const SrtRuntime& runtime, DescriptorValue& result) const {
    std::vector<DescriptorValue> results;
    EvaluateDescriptorSources(program, std::span {&source, 1}, runtime, results);
    result = results.front();
}

void SrtWalker::EvaluateDescriptorSources(const IrResourcePlan& program, std::span<const std::uint32_t> sources, const SrtRuntime& runtime, std::vector<DescriptorValue>& results) const {
    std::vector<std::uint32_t> ignored;
    std::vector<std::uint8_t> active;
    if (!Detail::EvaluateRuntimeSourcesImpl(program, sources, runtime, results, ignored, false, {}, active)) {
        throw std::runtime_error("SrtWalker::EvaluateDescriptorSources failed to evaluate descriptor sources: " + Detail::RuntimeSourceFailureReason());
    }
}

void SrtWalker::EvaluateRuntimeSources(const IrResourcePlan& program, std::span<const std::uint32_t> sources, const SrtRuntime& runtime, std::vector<DescriptorValue>& results, std::vector<std::uint32_t>& flat, std::span<const std::uint8_t> cleanFlatSlots, std::vector<std::uint8_t>& activeSources, std::vector<SrtReadPoison>* poison, std::uint32_t* nullRootReads) const {
    if (!Detail::EvaluateRuntimeSourcesImpl(program, sources, runtime, results, flat, true, cleanFlatSlots, activeSources, poison, nullRootReads)) {
        throw std::runtime_error("SrtWalker::EvaluateRuntimeSources failed to evaluate runtime sources: " + Detail::RuntimeSourceFailureReason());
    }
}

void SrtWalker::Walk(const IrResourcePlan& program, const SrtRuntime& runtime, std::vector<std::uint32_t>& flat) const {
    std::vector<DescriptorValue> ignored;
    std::vector<std::uint8_t> active;
    EvaluateRuntimeSources(program, {}, runtime, ignored, flat, {}, active);
}

}
