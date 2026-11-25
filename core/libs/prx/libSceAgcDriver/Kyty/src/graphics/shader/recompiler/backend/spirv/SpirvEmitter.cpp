#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	EXIT("SPIR-V validation failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     program.shader_hash, static_cast<unsigned>(program.stage), reason);
	std::abort();
}

// A compute wave64 program whose workgroup is one wave can run one guest lane per host invocation
// on two 32-wide subgroups instead of two lanes per invocation (one code copy instead of two).
// Its cross-half values (ballots, lane reads, branch votes, reductions) then go through workgroup
// memory between workgroup barriers, which needs every branch to be wave-uniform: guest branches
// are, as votes are exchanged and scalar values are computed alike in both halves. Excluded: what
// one half would compute differently from the other (clocks, coherent scalar loads that another
// wave may change between the halves' loads), elect-one-lane atomics (append/consume), and
// layouts the exchange cannot share (quad derivative groups, explicit-layout LDS, GDS).
bool CanSplitWave(const IR::Program& program, const ShaderStageInputInfo& input_info,
                  const Profile& profile) {
	if (!profile.wave_split || program.stage != ShaderType::Compute ||
	    program.dispatcher_fallback) {
		return false;
	}
	const auto& info = program.info;
	if (info.compute_derivatives || info.uses_gds || info.shared_int64_atomics) {
		return false;
	}
	const auto* cs = input_info.compute;
	if (cs == nullptr) {
		return false;
	}
	const auto threads = std::max(cs->threads_num[0], 1u) * std::max(cs->threads_num[1], 1u) *
	                     std::max(cs->threads_num[2], 1u);
	if (threads > 64u) {
		return false;
	}
	// The exchange takes 1 KiB of workgroup memory next to the guest's LDS.
	if (info.uses_lds && !cs->lds_storage && cs->lds_size_dwords * 4u > 31u * 1024u) {
		return false;
	}
	using Op = IR::ValueOpcode;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (op == Op::ReadRealtimeClock || op == Op::DataAppend || op == Op::DataConsume) {
				return false;
			}
			// The SMEM loads that the lane pair loads once for both halves.
			if (op == Op::ReadConstBuffer || op == Op::LoadAddressU32) {
				const auto index = inst.Flags<IR::MemoryFlags>().index;
				if (index >= program.memory_info.size()) {
					return false;
				}
				const auto& mem = program.memory_info[index];
				if (mem.coherent && (op == Op::ReadConstBuffer ||
				                     mem.kind == IR::ResourceKind::ScalarAddress)) {
					return false;
				}
			}
		}
	}
	return true;
}

} // namespace

std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t push_data_start_dword, const Profile& profile,
                                  bool emit_debug_info, uint32_t push_data_ceiling) {
	using namespace Emitter;

	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	IR::ValidateProgram(program);
	program.bindings = {.push_data_start_dword = push_data_start_dword};
	EmitterState state(program, input_info, profile);
	if (emit_debug_info) {
		char name[64];
		std::snprintf(name, sizeof(name), "rdna2-%016" PRIx64 "-block-byte-pc-plus-one",
		              program.shader_hash);
		state.debug_source = state.builder.AddString(name);
		state.builder.EnableNames();
	}
	const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
	state.lane_count =
	    workgroup != nullptr && program.wave_size == 64u && workgroup->host_subgroup_size == 32u
	        ? 2u
	        : 1u;
	if (state.lane_count == 2 && CanSplitWave(program, input_info, profile)) {
		state.lane_count = 1;
		state.wave_split = true;
	}
	DefineModule(state, push_data_ceiling);
	EmitProgram(state);
	state.builder.AddEntryPoint(ExecutionModelForStage(state.program.stage), state.main_func,
	                            "main", state.interface_variables);

	return state.builder.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
