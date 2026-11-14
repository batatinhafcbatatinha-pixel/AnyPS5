#include "Optimization/SrtWalker/SrtPlanBuilder.hpp"
#include "Optimization/SrtWalker.hpp"
#include "Optimization/SrtWalker/SrtDiagnostics.hpp"
#include "Optimization/SrtWalker/SrtInstructionPredicates.hpp"
#include "Optimization/SrtWalker/SrtRuntimeValidator.hpp"

#include "IntermediateRepresentation/IrBuilder.hpp"
#include "Recompiler.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace ShaderRecompiler::Detail {

void PlanBuilder::Run() {
    _program.Resources().srtReads.clear();
    _program.Metadata().dynamicReads.clear();
    // imagesOnly: a program that stores to or updates any buffer keeps every V# walked and bound (see
    // ResourceTracker ProgramWritesBuffers). With APS5_GPU_SRT=4 (global images) the T#s of sampled
    // images and all S#s are left to the shader too; images a storage, packed or by-element access
    // uses keep their CPU walk (ResourceTracker DirectImageHandles binds them directly).
    bool programWrites = false;
    if (_imagesOnly) FindShaderBased();
    if (_imagesOnly) {
        for (auto& block : _program.Blocks()) {
            for (const IrValue* inst : block->Instructions()) {
                const auto buffer = BufferAccessOf(inst->Opcode());
                const auto address = AddressOpcodeInfoOf(inst->Opcode()).access;
                if (buffer == BufferAccess::Write || buffer == BufferAccess::Atomic || address == AddressAccess::Write || address == AddressAccess::Atomic) programWrites = true;
            }
        }
    }
    // A program that writes any image keeps all its images and samplers walked and bound: a sampled
    // global image is a separate view made from guest memory, so a program reading back what it
    // stores (feedback textures, counters) would never see its stores (the mode-3 GPU hang, for
    // buffers, eva_d1; ResourceTracker::GlobalImages applies the same rule).
    // Compute programs keep their images walked too: a first use of a key misses the global map and
    // samples 0, and compute results the game reads back (streaming, visibility) must not see that
    // (the mode-4 save-load stall, eva_u1..u5). APS5_GPU_SRT_GLOBAL_COMPUTE=1 includes them.
    static const bool globalCompute = std::getenv("APS5_GPU_SRT_GLOBAL_COMPUTE") != nullptr;
    const bool globalImages = _imagesOnly && GpuSrtStages() >= 4u && !ProgramWritesImages(_program) && (globalCompute || _program.Resources().stage != IrShaderStage::Compute);
    // Same switch as ResourceTracker TakeGpuDescriptor: written V#s go through BDA too.
    static const bool gpuWrites = std::getenv("APS5_GPU_SRT_WRITES") != nullptr;
    // Programs that use images keep their V#s and raw SRT reads on the CPU walk (only their images
    // and samplers go GPU-side, mode 4): their GPU-time reads of SRT memory stalled the save load
    // (bisect eva_u14..u18; eva_u17 loads). APS5_GPU_SRT_IMAGE_PROGRAM_GPU_READS=1 reads them on the GPU.
    static const bool imageProgramGpuReads = std::getenv("APS5_GPU_SRT_IMAGE_PROGRAM_GPU_READS") != nullptr;
    const bool boundImageProgram = _imagesOnly && !imageProgramGpuReads && !SrtWalker::GpuSrtEligibleWithoutImages(_program);
    const auto& memoryInfo = _program.Resources().memoryInfo;
    const auto walkedImage = [&](const IrValue* handle) {
        for (const IrValue* use : handle->Uses()) {
            const auto info = ImageOpcodeInfoOf(use->Opcode());
            if (use->Opcode() == IrOpcode::ImageBvhIntersectRay || info.access == ImageAccess::None || use->ArgumentCount() == 0u || use->Argument(0)->Resolve() != handle) return true;
            if (info.resourceClass == ImageResourceClass::Storage) return true;
            const auto index = use->Flags<MemoryFlags>().index;
            if (index >= memoryInfo.size() || memoryInfo[index].imagePacked || memoryInfo[index].imageByElements != 0u) return true;
        }
        return false;
    };
    for (auto& block : _program.Blocks()) {
        for (IrValue* inst : block->Instructions()) {
            const auto op = inst->Opcode();
            if (op == IrOpcode::LoadAddressU32 || op == IrOpcode::ReadConstBuffer) {
                const auto flags = inst->Flags<MemoryFlags>();
                if (flags.index < _program.Resources().memoryInfo.size()) {
                    const auto kind = _program.Resources().memoryInfo[flags.index].kind;
                    const bool crosswired = (op == IrOpcode::LoadAddressU32 && kind == ResourceKind::ScalarBuffer) || (op == IrOpcode::ReadConstBuffer && kind == ResourceKind::ScalarAddress);
                    if (crosswired) {
                        Fail(_program.Resources(), flags.pc, std::string(IrOpcodeName(op)) + " has incompatible scalar memory metadata");
                    }
                }
            }
            // imagesOnly also plans the V# of every buffer a store or atomic goes through: written
            // buffers stay bound descriptors (BDA stores bypass resident copies of the memory).
            const auto writtenThrough = [](const IrValue* handle) {
                for (const IrValue* use : handle->Uses()) {
                    const auto access = BufferAccessOf(use->Opcode());
                    if (access == BufferAccess::Write || access == BufferAccess::Atomic) return true;
                }
                return false;
            };
            const bool walked = !_imagesOnly
                || (op == IrOpcode::GetImageResource && (!globalImages || walkedImage(inst) || DependsOnShaderBase(inst)))
                || (op == IrOpcode::GetSamplerResource && (!globalImages || DependsOnShaderBase(inst)))
                || (op == IrOpcode::GetBufferResource && ((!gpuWrites && (programWrites || writtenThrough(inst))) || DependsOnShaderBase(inst) || boundImageProgram));
            if (IsDescriptorHandle(op) && walked) {
                for (std::size_t index = 0; index < inst->ArgumentCount(); index++) {
                    Collect(inst->Argument(index), 0);
                }
            }
        }
    }
    const bool walkRaw = boundImageProgram;
    for (auto& block : _program.Blocks()) {
        // Debug aid: APS5_GPU_SRT_PLAN_RAW=1 still plans the immediate raw reads (flat slots, branch
        // conditions) in an imagesOnly plan.
        static const bool planRaw = std::getenv("APS5_GPU_SRT_PLAN_RAW") != nullptr;
        for (IrValue* inst : block->Instructions()) {
            // imagesOnly: only the raw reads of the shader's own code (s_getpc tables): GetShaderBase
            // is 0 on the GPU (SpirvModuleEmitter EmitGetShaderBase), so such reads stay walked.
            static const bool verifyRaw = std::getenv("APS5_GPU_SRT_VERIFY_RAW") != nullptr;
            if (_imagesOnly && !planRaw && !verifyRaw && !walkRaw && !DependsOnShaderBase(inst)) continue;
            if (inst->Opcode() == IrOpcode::LoadAddressU32 && IsRawRead(_program.Resources(), *inst) && inst->Argument(1)->Resolve()->HasImmediate() && RuntimeValidator(_program.Resources(), RuntimeValueType::Any).Run(inst)) {
                Collect(inst, inst->Flags<MemoryFlags>().pc);
            }
        }
    }
    PatchReads();
}

bool ProgramWritesImages(const IrProgram& program) {
    for (const auto& block : program.Blocks()) {
        for (const IrValue* inst : block->Instructions()) {
            const auto info = ImageOpcodeInfoOf(inst->Opcode());
            if (info.access == ImageAccess::Atomic || inst->Opcode() == IrOpcode::ImageWrite) return true;
        }
    }
    return false;
}

bool PlanBuilder::DependsOnShaderBase(const IrValue* value) const {
    return _shaderBased.contains(value->Resolve());
}

// The values computed from GetShaderBase (0 on the GPU), found once per plan by forward propagation
// to a fixed point (Phi back edges take a second pass).
void PlanBuilder::FindShaderBased() {
    _shaderBased.clear();
    for (bool changed = true; changed;) {
        changed = false;
        for (auto& block : _program.Blocks()) {
            for (const IrValue* inst : block->Instructions()) {
                const IrValue* value = inst->Resolve();
                if (_shaderBased.contains(value)) continue;
                bool based = value->Opcode() == IrOpcode::GetShaderBase;
                for (std::size_t argument = 0; !based && argument < value->ArgumentCount(); ++argument) {
                    based = value->Argument(argument) != nullptr && _shaderBased.contains(value->Argument(argument)->Resolve());
                }
                if (based) changed = _shaderBased.insert(value).second || changed;
            }
        }
    }
}

void PlanBuilder::Collect(IrValue* raw, std::uint32_t usePc) {
    struct Frame {
        IrValue* inst = nullptr;
        std::size_t next = 0;
    };
    std::vector<Frame> stack;
    const auto enter = [&](IrValue* candidate) {
        IrValue* inst = candidate->Resolve();
        if (inst->Opcode() == IrOpcode::Void) {
            return;
        }
        if (const auto cycle = _visitingIndex.find(inst); cycle != _visitingIndex.end()) {
            const auto containsPhi = std::any_of(_visiting.begin() + static_cast<std::ptrdiff_t>(cycle->second), _visiting.end(), [](IrValue* candidate) { return candidate->Opcode() == IrOpcode::Phi; });
            if (containsPhi) {
                return;
            }
            Fail(_program.Resources(), usePc, "cyclic typed planning value " + std::string(IrOpcodeName(inst->Opcode())) + " without a phi");
        }
        if (_visited.contains(inst)) {
            return;
        }
        _visitingIndex.emplace(inst, _visiting.size());
        _visiting.push_back(inst);
        stack.push_back({inst, 0});
    };
    enter(raw);
    while (!stack.empty()) {
        auto& frame = stack.back();
        if (frame.next < frame.inst->ArgumentCount()) {
            IrValue* argument = frame.inst->Argument(frame.next);
            frame.next++;
            enter(argument);
            continue;
        }
        IrValue* inst = frame.inst;
        stack.pop_back();
        _visiting.pop_back();
        _visitingIndex.erase(inst);
        _visited.insert(inst);
        Record(inst);
    }
}

void PlanBuilder::Record(IrValue* inst) {
    if (!IsRawRead(_program.Resources(), *inst)) {
        return;
    }
    IrValue* offset = inst->Argument(1)->Resolve();
    const auto foldable = offset->HasImmediate() && offset->Type() == IrType::U32 && RuntimeValidator(_program.Resources(), RuntimeValueType::Any).Run(inst);
    if (!foldable) {
        if (std::find(_program.Metadata().dynamicReads.begin(), _program.Metadata().dynamicReads.end(), inst) == _program.Metadata().dynamicReads.end()) {
            _program.Metadata().dynamicReads.push_back(inst);
        }
        return;
    }
    for (std::uint32_t slot = 0; slot < _program.Resources().srtReads.size(); slot++) {
        if (EquivalentValue(_program.Resources(), inst, _program.Resources().srtReads[slot].value)) {
            _patches.push_back({inst, slot, false});
            return;
        }
    }
    const auto slot = static_cast<std::uint32_t>(_program.Resources().srtReads.size());
    _program.Resources().srtReads.push_back({inst, slot});
    _patches.push_back({inst, slot, true});
}

void PlanBuilder::PatchReads() {
    IrBuilder builder(_program);
    for (const auto& patch : _patches) {
        IrBlock* block = patch.inst->Parent();
        if (block == nullptr) {
            throw std::runtime_error("SrtWalker::BuildPlan patched value has no parent block");
        }
        builder.SetInsertionPoint(*block);
        IrValue& resource = _program.CreateValue(IrOpcode::GetSrtResource, IrType::SrtResource);
        block->InsertInstructionBefore(patch.inst, &resource);
        IrValue& slotConstant = builder.Constant(patch.slot);
        IrValue& flat = _program.CreateValue(IrOpcode::ReadConst, IrType::U32);
        flat.AddArgument(&resource);
        flat.AddArgument(&slotConstant);
        block->InsertInstructionBefore(patch.inst, &flat);
        patch.inst->ReplaceAllUsesWith(&flat);
        // Debug aid: APS5_GPU_SRT_VERIFY_RAW=1 keeps the GPU's own read of a walked word in a GPU-SRT
        // program and reports a value differing from the CPU walk's as a descriptor fault (both
        // words and the instruction's pc), to find memory the GPU sees differently from the capture.
        static const bool verifyRaw = std::getenv("APS5_GPU_SRT_VERIFY_RAW") != nullptr;
        if (verifyRaw && _imagesOnly && patch.inst->Type() == IrType::U32 && patch.inst->Opcode() == IrOpcode::LoadAddressU32) {
            auto& instructions = block->Instructions();
            auto next = std::find(instructions.begin(), instructions.end(), patch.inst);
            if (next != instructions.end() && ++next != instructions.end()) {
                IrValue* position = *next;
                IrValue& differs = _program.CreateValue(IrOpcode::INotEqual32, IrType::Bool);
                differs.AddArgument(&flat);
                differs.AddArgument(patch.inst);
                block->InsertInstructionBefore(position, &differs);
                IrValue& pc = builder.Constant(patch.inst->Flags<MemoryFlags>().pc);
                IrValue& fault = _program.CreateValue(IrOpcode::DescriptorFault, IrType::Void);
                fault.AddArgument(&differs);
                fault.AddArgument(&flat);
                fault.AddArgument(patch.inst);
                fault.AddArgument(&pc);
                block->InsertInstructionBefore(position, &fault);
                _program.Resources().info.usesFaultBuffer = true;
            }
        }
        for (auto& info : _program.Metadata().blockInfo) {
            if (info.condition != nullptr && info.condition->Resolve() == patch.inst) {
                info.condition = &flat;
            }
            if (info.indirectTarget != nullptr && info.indirectTarget->Resolve() == patch.inst) {
                info.indirectTarget = &flat;
            }
        }
        if (patch.keep) {
            const auto memory = patch.inst->Flags<MemoryFlags>().index;
            if (memory < _program.Resources().memoryInfo.size()) {
                _program.Resources().memoryInfo[memory].planningOnly = true;
            }
            const auto& irVal = builder.Emit(IrOpcode::ReferenceU32, IrType::Void, {patch.inst});
        }
    }
}

}
