#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include <bit>
#include <cstdlib>
#include <utility>
#include <iterator>
#include <initializer_list>
#include <vector>

namespace ShaderRecompiler {

namespace {

// (u64 address, u32 bytes, u32 instruction) -> u64 device address of the range holding all the bytes,
std::uint32_t DefineBdaLookup(SpirvEmitterState& state, const char* name, bool recordFaults, std::uint32_t permission = BdaAbi::Read, bool rangeResult = false) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, name);
    const auto rangeType = rangeResult ? state.module.Type(spv::OpTypeVector, u64, 3u) : 0u;
    const auto resultType = rangeResult ? rangeType : u64;
    const auto zero = rangeResult ? state.module.Constant(spv::OpConstantComposite, rangeType, BdaConstant(state, 0u), BdaConstant(state, 0u), BdaConstant(state, 0u)) : BdaConstant(state, 0u);
    state.module.AddFunction(spv::OpFunction, resultType, function, rangeResult ? spv::FunctionControlMaskNone : spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, resultType, u64, u32, u32));
    const auto address = state.module.AllocateId();
    const auto bytes = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto outer = state.module.AllocateId();
    const auto outerBody = state.module.AllocateId();
    const auto outerContinue = state.module.AllocateId();
    const auto outerMerge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, outer);
    EmitLabel(state, outer);
    state.module.AddFunction(spv::OpLoopMerge, outerMerge, outerContinue, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranch, outerBody);
    EmitLabel(state, outerBody);
    std::vector<std::uint32_t> exits{spv::OpPhi, resultType, state.module.AllocateId()};
    const auto leave = [&](std::uint32_t value) {
        exits.push_back(value);
        exits.push_back(state.currentLabel);
        state.module.AddFunction(spv::OpBranch, outerMerge);
    };
    const auto leaveIf = [&](std::uint32_t condition, std::uint32_t reason) {
        const auto failed = state.module.AllocateId();
        const auto next = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, condition, failed, next);
        EmitLabel(state, failed);
        if (recordFaults) RecordBdaFaultReason(state, address, bytes, instruction, reason);
        leave(zero);
        EmitLabel(state, next);
    };
    const auto fail = [&](std::uint32_t condition, BdaAbi::FaultReason reason) { leaveIf(condition, recordFaults ? constant(static_cast<std::uint32_t>(reason)) : 0u); };
    const auto failFirst = [&](std::initializer_list<std::pair<std::uint32_t, BdaAbi::FaultReason>> checks) {
        auto reason = constant(0u);
        for (auto check = std::rbegin(checks); check != std::rend(checks); ++check) {
            reason = Select(state, u32, check->first, constant(static_cast<std::uint32_t>(check->second)), reason);
        }
        leaveIf(binary(spv::OpINotEqual, boolean, reason, constant(0u)), reason);
    };
    std::vector<std::uint32_t> pendingMerges;
    const auto length = state.module.AllocateId();
    state.module.AddFunction(spv::OpArrayLength, u32, length, state.bdaPagetableVariable, 0u);
    fail(binary(spv::OpULessThan, boolean, length, constant(4)), BdaAbi::FaultReason::InvalidTable);
    const auto count = BdaLoadWord(state, constant(1));
    // Header word 3 bit 0 (BdaAbi::CacheRootFlag): the device address of the buffer cache's page table
    // root (Graphics BufferCache.hpp) follows the last range as two words.
    const auto flags = BdaLoadWord(state, constant(3));
    const auto cached = binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, flags, constant(BdaAbi::CacheRootFlag)), constant(0));
    const auto available = binary(spv::OpISub, u32, binary(spv::OpISub, u32, length, constant(4)), Select(state, u32, cached, constant(2), constant(0)));
    const auto end = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
    failFirst({
        {binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(0)), constant(BdaAbi::Version)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(2)), constant(sizeof(BdaAbi::Range))), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, flags, constant(~BdaAbi::CacheRootFlag)), constant(0)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, available, constant(7)), constant(0)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpINotEqual, boolean, count, binary(spv::OpShiftRightLogical, u32, available, constant(3))), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpIEqual, boolean, bytes, constant(0)), BdaAbi::FaultReason::Overflow},
        {binary(spv::OpULessThanEqual, boolean, end, address), BdaAbi::FaultReason::Overflow},
    });
    // Debug aids to bisect the GPU-SRT lookup additions: APS5_GPU_SRT_NO_TABLE=1 (no page-table fast
    // path), APS5_GPU_SRT_NO_ACCESSED=1 (no accessed-page notes).
    static const bool tableLookups = std::getenv("APS5_GPU_SRT_NO_TABLE") == nullptr;
    static const bool accessedNotes = std::getenv("APS5_GPU_SRT_NO_ACCESSED") == nullptr;
    if (tableLookups && permission == BdaAbi::Read && state.program.Resources().gpuSrt) {
        // Read-only fast path: root[A >> RootShift] -> leaf, leaf[(A & leaf mask) >> PageShift] -> page,
        // device = page + (A & page mask), when the page is resident and the bytes stay in it; any 0
        // entry falls through to the range search below. It skips the range checks (permission,
        // InvalidTable): the buffer cache's table is trusted, and only read lookups take it.
        const auto physicalU64 = TypePointer(state, spv::StorageClassPhysicalStorageBuffer, u64);
        const auto loadEntry = [&](std::uint32_t at) {
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, physicalU64, pointer, at);
            const auto value = state.module.AllocateId();
            state.module.AddFunction(spv::OpLoad, u64, value, pointer, spv::MemoryAccessAlignedMask, 8u);
            return value;
        };
        const auto rootIndex = binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::CacheLeafShift));
        const auto inRoot = binary(spv::OpULessThan, boolean, rootIndex, BdaConstant(state, BdaAbi::CacheRoots));
        const auto pageOffset = binary(spv::OpBitwiseAnd, u64, address, BdaConstant(state, (std::uint64_t{1} << BdaAbi::CachePageShift) - 1u));
        const auto inPage = binary(spv::OpULessThanEqual, boolean, binary(spv::OpIAdd, u64, pageOffset, Unary(state, spv::OpUConvert, u64, bytes)), BdaConstant(state, std::uint64_t{1} << BdaAbi::CachePageShift));
        const auto tryTable = binary(spv::OpLogicalAnd, boolean, binary(spv::OpLogicalAnd, boolean, cached, inRoot), inPage);
        const auto zero64 = BdaConstant(state, 0u);
        const auto page = EmitValueIfElse(state, tryTable, u64, [&] {
            const auto root = BdaLoadAddress(state, binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, count, constant(8))));
            const auto leaf = EmitValueIfElse(state, binary(spv::OpINotEqual, boolean, root, zero64), u64, [&] {
                return loadEntry(binary(spv::OpIAdd, u64, root, binary(spv::OpShiftLeftLogical, u64, rootIndex, BdaConstant(state, 3u))));
            }, [&] { return zero64; });
            return EmitValueIfElse(state, binary(spv::OpINotEqual, boolean, leaf, zero64), u64, [&] {
                const auto leafIndex = binary(spv::OpShiftRightLogical, u64, binary(spv::OpBitwiseAnd, u64, address, BdaConstant(state, (std::uint64_t{1} << BdaAbi::CacheLeafShift) - 1u)), BdaConstant(state, BdaAbi::CachePageShift));
                return loadEntry(binary(spv::OpIAdd, u64, leaf, binary(spv::OpShiftLeftLogical, u64, leafIndex, BdaConstant(state, 3u))));
            }, [&] { return zero64; });
        }, [&] { return zero64; });
        const auto hit = state.module.AllocateId();
        const auto miss = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, miss, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, binary(spv::OpINotEqual, boolean, page, zero64), hit, miss);
        EmitLabel(state, hit);
        if (rangeResult) {
            const auto pageBegin = binary(spv::OpISub, u64, address, pageOffset);
            const auto range = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, rangeType, range, pageBegin, binary(spv::OpIAdd, u64, pageBegin, BdaConstant(state, std::uint64_t{1} << BdaAbi::CachePageShift)), page);
            leave(range);
        } else {
            leave(binary(spv::OpIAdd, u64, page, pageOffset));
        }
        EmitLabel(state, miss);
    }
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    const auto entryLabel = state.currentLabel;
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto lower = state.module.AllocateId();
    const auto upper = state.module.AllocateId();
    const auto newLow = state.module.AllocateId();
    const auto newHigh = state.module.AllocateId();
    state.module.AddFunction(spv::OpPhi, u32, lower, constant(0), entryLabel, newLow, continuation);
    state.module.AddFunction(spv::OpPhi, u32, upper, count, entryLabel, newHigh, continuation);
    const auto search = binary(spv::OpULessThan, boolean, lower, upper);
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, search, body, merge);
    EmitLabel(state, body);
    const auto midpoint = binary(spv::OpIAdd, u32, lower, binary(spv::OpShiftRightLogical, u32, binary(spv::OpISub, u32, upper, lower), constant(1)));
    const auto entry = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, midpoint, constant(8)));
    const auto before = binary(spv::OpULessThan, boolean, address, BdaLoadAddress(state, entry));
    state.module.AddFunction(spv::OpSelect, u32, newLow, before, lower, binary(spv::OpIAdd, u32, midpoint, constant(1)));
    state.module.AddFunction(spv::OpSelect, u32, newHigh, before, midpoint, upper);
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto index = lower;
    fail(binary(spv::OpIEqual, boolean, index, constant(0)), BdaAbi::FaultReason::Unmapped);
    const auto selected = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, binary(spv::OpISub, u32, index, constant(1)), constant(8)));
    const auto at = [&](std::uint32_t offset) { return binary(spv::OpIAdd, u32, selected, constant(offset)); };
    const auto begin = BdaLoadAddress(state, selected);
    const auto finish = BdaLoadAddress(state, at(2));
    const auto base = BdaLoadAddress(state, at(4));
    const auto permissions = BdaLoadWord(state, at(6));
    const auto offset = binary(spv::OpISub, u64, address, begin);
    const auto result = binary(spv::OpIAdd, u64, base, offset);
    const auto deviceEnd = binary(spv::OpIAdd, u64, result, Unary(state, spv::OpUConvert, u64, bytes));
    failFirst({
        {binary(spv::OpINotEqual, boolean, BdaLoadWord(state, at(7)), constant(0)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpUGreaterThanEqual, boolean, begin, finish), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpUGreaterThan, boolean, end, finish), BdaAbi::FaultReason::Unmapped},
        {binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, permissions, constant(permission)), constant(permission)), BdaAbi::FaultReason::Permission},
        {binary(spv::OpIEqual, boolean, base, BdaConstant(state, 0)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpULessThan, boolean, result, base), BdaAbi::FaultReason::Overflow},
        {binary(spv::OpULessThanEqual, boolean, deviceEnd, result), BdaAbi::FaultReason::Overflow},
    });
    if (accessedNotes && permission == BdaAbi::Read && state.program.Resources().gpuSrt && state.faultBufferVariable != 0) {
        // The page table missed (or is absent): tell Graphics this page is used (cache flag only).
        // The u32 key holds addresses below 2^(32 + CachePageShift); higher ones are not noted. The
        // count cannot pass AccessedPageSlots: a claim is a successful CAS on an empty slot.
        const auto keyable = binary(spv::OpULessThan, boolean, address, BdaConstant(state, std::uint64_t{1} << (32u + BdaAbi::CachePageShift)));
        EmitIfCondition(state, binary(spv::OpLogicalAnd, boolean, cached, keyable), [&] {
            const auto key = binary(spv::OpIAdd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::CachePageShift))), constant(1u));
            const auto hash = binary(spv::OpShiftRightLogical, u32, binary(spv::OpIMul, u32, key, constant(0x9e3779b1u)), constant(32u - static_cast<std::uint32_t>(std::countr_zero(BdaAbi::AccessedPageSlots))));
            const auto scope = constant(spv::ScopeDevice);
            const auto relaxed = constant(spv::MemorySemanticsMaskNone);
            for (std::uint32_t probe = 0; probe < BdaAbi::AccessedPageProbes; ++probe) {
                // Unrolled probes, each skipped once an earlier one found or claimed the key.
                const auto slot = binary(spv::OpBitwiseAnd, u32, binary(spv::OpIAdd, u32, hash, constant(probe)), constant(BdaAbi::AccessedPageSlots - 1u));
                const auto word = binary(spv::OpIAdd, u32, constant(BdaAbi::AccessedSlotsWord), slot);
                const auto previous = state.module.AllocateId();
                state.module.AddFunction(spv::OpAtomicCompareExchange, u32, previous, BdaWord(state, state.faultBufferVariable, word), scope, relaxed, relaxed, key, constant(0u));
                EmitIfCondition(state, binary(spv::OpIEqual, boolean, previous, constant(0u)), [&] {
                    const auto claim = state.module.AllocateId();
                    state.module.AddFunction(spv::OpAtomicIAdd, u32, claim, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::AccessedCountWord)), scope, relaxed, constant(1u));
                    EmitIfCondition(state, binary(spv::OpULessThan, boolean, claim, constant(BdaAbi::AccessedPageSlots)), [&] {
                        state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, binary(spv::OpIAdd, u32, constant(BdaAbi::AccessedListWord), claim)), scope, relaxed, slot);
                    });
                });
                // Stop probing when this slot holds the key now (claimed here or earlier).
                const auto settled = binary(spv::OpLogicalOr, boolean, binary(spv::OpIEqual, boolean, previous, constant(0u)), binary(spv::OpIEqual, boolean, previous, key));
                if (probe + 1u < BdaAbi::AccessedPageProbes) {
                    const auto next = state.module.AllocateId();
                    const auto done = state.module.AllocateId();
                    state.module.AddFunction(spv::OpSelectionMerge, done, spv::SelectionControlMaskNone);
                    state.module.AddFunction(spv::OpBranchConditional, settled, done, next);
                    EmitLabel(state, next);
                    pendingMerges.push_back(done);
                }
            }
            for (auto merge = pendingMerges.rbegin(); merge != pendingMerges.rend(); ++merge) {
                state.module.AddFunction(spv::OpBranch, *merge);
                EmitLabel(state, *merge);
            }
            pendingMerges.clear();
        });
    }
    if (rangeResult) {
        const auto last = binary(spv::OpUGreaterThanEqual, boolean, index, count);
        const auto next = BdaLoadAddress(state, binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, Select(state, u32, last, binary(spv::OpISub, u32, index, constant(1)), index), constant(8))));
        const auto bound = Select(state, u64, binary(spv::OpLogicalOr, boolean, last, binary(spv::OpULessThan, boolean, finish, next)), finish, next);
        const auto range = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, rangeType, range, begin, bound, base);
        leave(range);
    } else {
        leave(result);
    }
    EmitLabel(state, outerContinue);
    state.module.AddFunction(spv::OpBranch, outer);
    EmitLabel(state, outerMerge);
    state.module.AddFunction(exits);
    state.module.AddFunction(spv::OpReturnValue, exits[2]);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

}

static std::uint32_t DefineBdaNoteWrite(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "note_bda_write");
    state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlMaskNone, state.module.Type(spv::OpTypeFunction, voidType, TypeScalarU64(state)));
    const auto address = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, TypeScalarU64(state), address);
    EmitLabel(state, state.module.AllocateId());
    const auto done = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, boolean), done, spv::StorageClassFunction);
    const auto counter = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, u32), counter, spv::StorageClassFunction);
    state.module.AddFunction(spv::OpStore, done, ConstantBool(state, false));
    state.module.AddFunction(spv::OpStore, counter, constant(0u));
    const auto u64 = TypeScalarU64(state);
    const auto key = binary(spv::OpIAdd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::WrittenChunkShift))), constant(1));
    const auto page = binary(spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::WrittenPageShift))), constant(BdaAbi::WrittenChunkPages - 1u));
    const auto bit = binary(spv::OpShiftLeftLogical, u32, constant(1u), page);
    const auto hash = binary(spv::OpShiftRightLogical, u32, binary(spv::OpIMul, u32, key, constant(0x9e3779b1u)), constant(32u - static_cast<std::uint32_t>(std::countr_zero(BdaAbi::WrittenPageSlots))));
    const auto scope = constant(spv::ScopeDevice);
    const auto relaxed = constant(spv::MemorySemanticsMaskNone);
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto probe = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, u32, probe, counter);
    const auto found = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, found, done);
    const auto searching = binary(spv::OpLogicalAnd, boolean, Unary(state, spv::OpLogicalNot, boolean, found), binary(spv::OpULessThan, boolean, probe, constant(BdaAbi::WrittenPageProbes)));
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, searching, body, merge);
    EmitLabel(state, body);
    const auto slot = binary(spv::OpBitwiseAnd, u32, binary(spv::OpIAdd, u32, hash, probe), constant(BdaAbi::WrittenPageSlots - 1u));
    const auto keyWord = binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenSlotsWord), binary(spv::OpShiftLeftLogical, u32, slot, constant(1u)));
    const auto current = state.module.AllocateId();
    state.module.AddFunction(spv::OpAtomicLoad, u32, current, BdaWord(state, state.faultBufferVariable, keyWord), scope, relaxed);
    const auto same = binary(spv::OpIEqual, boolean, current, key);
    state.module.AddFunction(spv::OpStore, done, same);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, same), [&] {
        const auto previous = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicCompareExchange, u32, previous, BdaWord(state, state.faultBufferVariable, keyWord), scope, relaxed, relaxed, key, constant(0u));
        const auto claimed = binary(spv::OpIEqual, boolean, previous, constant(0u));
        state.module.AddFunction(spv::OpStore, done, binary(spv::OpLogicalOr, boolean, claimed, binary(spv::OpIEqual, boolean, previous, key)));
        EmitIfCondition(state, claimed, [&] {
            const auto index = state.module.AllocateId();
            state.module.AddFunction(spv::OpAtomicIAdd, u32, index, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenCountWord)), scope, relaxed, constant(1u));
            state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenListWord), index)), scope, relaxed, slot);
        });
    });
    const auto holds = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, holds, done);
    EmitIfCondition(state, holds, [&] {
        const auto maskWord = binary(spv::OpIAdd, u32, keyWord, constant(1u));
        const auto mask = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicLoad, u32, mask, BdaWord(state, state.faultBufferVariable, maskWord), scope, relaxed);
        EmitIfCondition(state, binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u32, mask, bit), constant(0u)), [&] {
            state.module.AddFunction(spv::OpAtomicOr, u32, state.module.AllocateId(), BdaWord(state, state.faultBufferVariable, maskWord), scope, relaxed, bit);
        });
    });
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpStore, counter, binary(spv::OpIAdd, u32, probe, constant(1u)));
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto noted = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, noted, done);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, noted), [&] {
        state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenOverflowWord)), scope, relaxed, constant(1u));
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

void DefineBdaAnchorFunction(SpirvEmitterState& state) {
    if (state.bdaAnchorFunction == 0) state.bdaAnchorFunction = DefineBdaLookup(state, "bda_anchor_range", false, BdaAbi::Read, true);
}

void DefineGetBdaPointer(SpirvEmitterState& state) {
    if (state.faultBufferVariable != 0) DefineBdaFaultFunction(state);
    if (!state.program.Info().usesDma) return;
    state.bdaPointerFunction = DefineBdaLookup(state, "get_bda_pointer", true);
    if (!BdaByteReadsForced()) state.bdaProbeFunction = DefineBdaLookup(state, "probe_bda_pointer", false);
    DefineBdaDwordReadFunctions(state);
    DefineBdaSpanReadFunctions(state);
    CollectBdaAnchors(state);
    if (!state.bdaAnchors.empty() || state.program.Resources().gpuSrt) DefineBdaAnchorFunction(state);
    if (state.program.Info().bdaWrites) {
        state.bdaWritePointerFunction = DefineBdaLookup(state, "get_bda_write_pointer", true, BdaAbi::Write);
        if (!BdaByteReadsForced()) state.bdaWriteProbeFunction = DefineBdaLookup(state, "probe_bda_write_pointer", false, BdaAbi::Write);
        state.bdaAtomicPointerFunction = DefineBdaLookup(state, "get_bda_atomic_pointer", true, BdaAbi::Read | BdaAbi::Write);
        state.bdaNoteWriteFunction = DefineBdaNoteWrite(state);
        DefineBdaByteWriteFunctions(state);
    }
}

}
