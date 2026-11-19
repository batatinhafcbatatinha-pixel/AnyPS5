#include "graphics/shader/recompiler/ir/passes/RayQueryTraversal.h"

#include "common/logging/log.h"

#include <algorithm>
#include <array>
#include <bit>
#include <functional>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

// The descriptor of traversals over 64-bit node pointers: base 0, maximum size, sorted boxes and
// triangle barycentrics.
constexpr std::array<uint32_t, 4> TraversalDescriptor {0u, 0x80000000u, 0xffffffffu, 0x810003ffu};
constexpr uint32_t                One = 0x3f800000u;
// Bit (u | v << 2) is set for the remaps of a triangle node's barycentric vertices u and v, each
// 3 for 0, that order the triangle's vertices oddly.
constexpr uint32_t OddRemaps = 0x2942u;
// Dword 3 of a BVH T# selects barycentric triangle results with bit 24.
constexpr uint32_t Barycentrics = 1u << 24;

// Byte offsets in an instance record: BLAS header, root link and flags. Records whose BLAS is a
// raw address at +0 hold the instance mask in flags bits 8..15; records whose BLAS is a node
// pointer (address >> 3 under flag bits) hold it in bits 24..31.
constexpr uint32_t InstanceBlas = 0, InstanceRoot = 56, InstanceFlags = 60;

// How a BLAS address is built from two dwords of an instance record: (low, high & mask) << shift.
// A raw address has mask ~0 and shift 0.
struct BlasLayout {
	uint32_t offset = InstanceBlas; // Of the low dword.
	uint32_t mask   = ~0u;
	uint32_t shift  = 0;

	bool operator==(const BlasLayout&) const = default;

	// The byte of the instance flags that holds the instance mask.
	uint32_t MaskByte() const { return shift != 0 ? 3u : 1u; }
};

// The line of the first (innermost) condition a matcher failed on, for the log of traversals left
// to emulation.
thread_local int t_fail_line = 0;
bool             Fail(int line) {
	if (t_fail_line == 0) t_fail_line = line;
	return false;
}

Inst* Producer(Value value, ValueOpcode opcode) {
	auto* inst = value.Resolve().TryInstruction();
	return inst != nullptr && inst->GetOpcode() == opcode ? inst : nullptr;
}

std::optional<uint32_t> Immediate(Value value) {
	value = value.Resolve();
	if (!value.IsImmediate() || value.GetType() != Type::U32) return std::nullopt;
	return value.U32();
}

// The flattened SRT slot that `value` reads, if any.
std::optional<uint32_t> Slot(Value value) {
	const auto* read = Producer(value, ValueOpcode::ReadConst);
	return read != nullptr ? Immediate(read->Arg(1)) : std::nullopt;
}

size_t Incoming(const Inst& phi, const Block* from) {
	for (size_t index = 0; index < phi.NumArgs(); ++index) {
		if (phi.PhiBlock(index) == from) return index;
	}
	return phi.NumArgs();
}

// What a value the loop carries can be after an iteration: Select arms and phis, up to the header.
void Collect(Value value, const Block* header, std::unordered_set<const Inst*>& reached) {
	const auto* inst = value.Resolve().TryInstruction();
	if (inst == nullptr || !reached.insert(inst).second) return;
	if (inst->GetOpcode() == ValueOpcode::SelectU32) {
		Collect(inst->Arg(1), header, reached);
		Collect(inst->Arg(2), header, reached);
	} else if (inst->GetOpcode() == ValueOpcode::Phi && inst->Parent() != header) {
		for (size_t index = 0; index < inst->NumArgs(); ++index) {
			Collect(inst->Arg(index), header, reached);
		}
	}
}

// Read other than as a phi operand or the value a Select keeps.
bool DirectlyUsed(const Inst& inst) {
	return std::ranges::any_of(inst.Uses(), [](const Use& use) {
		const auto op = use.user->GetOpcode();
		return op != ValueOpcode::Phi && (op != ValueOpcode::SelectU32 || use.operand != 2);
	});
}

bool GlobalLoad(const Program& program, const Inst& inst, uint32_t offset) {
	if (inst.GetOpcode() != ValueOpcode::LoadAddressU32) return false;
	const auto& memory = program.memory_info[inst.Flags<MemoryFlags>().index];
	return memory.kind == ResourceKind::Global && memory.address_is_full && memory.offset == offset;
}

void Redirect(Inst& inst, Value replacement, const Inst& keep) {
	const auto uses = inst.Uses();
	for (const auto& use: uses) {
		if (use.user != &keep) use.user->SetArg(use.operand, replacement);
	}
}

struct Inserter {
	Block*          block;
	Block::iterator at;

	Value operator()(ValueOpcode opcode, std::initializer_list<Value> args,
	                 uint64_t flags = 0) const {
		auto& inst = *block->PrependNewInst(at, opcode, args, flags);
		return TypeOf(opcode) == Type::Void ? Value {} : Value(&inst);
	}

	// A copy of `inst` over `args`.
	Value operator()(const Inst& inst, std::span<const Value> args) const {
		auto& copy = *block->Instructions().emplace(at, inst.GetOpcode(), inst.Flags<uint64_t>());
		copy.SetParent(block);
		for (size_t arg = 0; arg < args.size(); ++arg) {
			copy.SetArg(arg, args[arg]);
		}
		return Value(&copy);
	}
};

Block::iterator Position(Inst& inst) {
	return std::ranges::find_if(*inst.Parent(), [&](const Inst& other) { return &other == &inst; });
}

size_t BlockIndex(const Program& program, const Block* block) {
	return static_cast<size_t>(std::ranges::find(program.blocks, block) - program.blocks.begin());
}

// The innermost loop around a block: the nearest header before it whose latch is at or after it.
Block* EnclosingLoop(const Program& program, const Block* block) {
	const auto at = BlockIndex(program, block);
	for (auto header = at; header-- > 0;) {
		const auto& term = program.blocks[header]->terminator;
		if (term.loop_header && term.continue_block != nullptr &&
		    BlockIndex(program, term.continue_block) >= at) {
			return program.blocks[header];
		}
	}
	return nullptr;
}

// A loop entered from one preheader: its header through its latch.
struct Loop {
	Block*                  preheader = nullptr;
	Block*                  header    = nullptr;
	Block*                  latch     = nullptr;
	std::span<Block* const> loop;
};

struct Traversal: Loop {
	Inst*                bvh    = nullptr;
	Inst*                exec   = nullptr; // The intersection's lanes.
	Inst*                extent = nullptr; // The closest committed t.
	Inst*                vote   = nullptr; // The latch's vote on whether lanes go on.
	std::vector<Inst*>   lanes;            // Phis that carry the voted lanes.
	std::array<Value, 7> ray {};           // World origin xyz, direction xyz, t_max.
	// Words 0 and 1 of the current prim record, the current instance address and flags; a nested
	// walk carries only the record words.
	std::array<Inst*, 5> state {};
	// A flat walk's loads of prim record words 0 and 1 and of instance flags.
	std::array<const Inst*, 3> loads {};
	// A nested walk's instance address and the loads of its BLAS words, defined before its loop,
	// its prim record load and the loop's scalar loads whose address they give.
	std::array<Inst*, 4>    instance {};
	BlasLayout              blas;
	// The values the candidate's replace: the index of each in instance address low and high, BLAS
	// address low and high, and the raw BLAS dwords.
	std::vector<std::pair<Inst*, uint32_t>> swaps;
	Inst*                   record = nullptr;
	std::vector<Inst*>      scalar;
	std::array<uint32_t, 2> tlas {};
	uint32_t                cull_mask = 0;
};

Value Entry(const Loop& t, const Inst& phi) {
	return phi.Arg(Incoming(phi, t.preheader));
}

Value Back(const Loop& t, const Inst& phi) {
	return phi.Arg(Incoming(phi, t.latch));
}

Inst* HeaderPhi(const Loop& t, Value value) {
	auto* inst = Producer(value, ValueOpcode::Phi);
	return inst != nullptr && inst->Parent() == t.header ? inst : nullptr;
}

bool Inside(const Loop& t, const Block* block) {
	return std::ranges::find(t.loop, block) != t.loop.end();
}

// Whether `header` heads a loop entered from one preheader, which `l` then holds.
bool Enter(const Program& program, Block* header, Loop& l) {
	if (header == nullptr || header->ImmPredecessors().size() != 2) return Fail(__LINE__);
	const auto entries = header->ImmPredecessors();
	l.header           = header;
	l.latch            = header->terminator.continue_block;
	l.preheader        = entries[0] == l.latch ? entries[1] : entries[0];
	l.loop             = std::span(program.blocks.begin() + BlockIndex(program, header),
	                               program.blocks.begin() + BlockIndex(program, l.latch) + 1);
	return std::ranges::find(entries, l.latch) != entries.end();
}

// The innermost loop around an intersection with extent `extent`, entered from one preheader, whose
// latch votes on the lanes the intersection runs on. The query's iterations skip the box path,
// which pushes intersection results, and run the rest on stale state: the loop must have no other
// side effects and leave only at the latch.
bool MatchLoop(const Program& program, Inst& bvh, Value extent, Traversal& t) {
	if (!std::ranges::all_of(bvh.Uses(), [](const Use& use) {
		    return use.user->GetOpcode() == ValueOpcode::CompositeExtractU32x4 &&
			       Immediate(use.user->Arg(1)) < 4u;
	    })) {
		return Fail(__LINE__);
	}
	if (!Enter(program, EnclosingLoop(program, bvh.Parent()), t) ||
	    t.header->terminator.kind != CFG::TerminatorKind::Branch ||
	    t.header->terminator.true_block == nullptr) {
		return Fail(__LINE__);
	}
	t.bvh    = &bvh;
	t.exec   = HeaderPhi(t, bvh.Arg(2));
	t.extent = HeaderPhi(t, extent);
	if (t.exec == nullptr || t.extent == nullptr) return Fail(__LINE__);

	// The latch's code goes before the vote, which must follow what the back edge carries.
	const auto lanes = Entry(t, *t.exec);
	const auto alive = Back(t, *t.exec);
	for (auto& inst: *t.header) {
		if (inst.GetOpcode() != ValueOpcode::Phi || inst.GetType() != Type::U1 ||
		    Back(t, inst) != alive) {
			continue;
		}
		if (Entry(t, inst) != lanes) return Fail(__LINE__);
		t.lanes.push_back(&inst);
	}
	const auto& term = t.latch->terminator;
	t.vote           = Producer(t.latch->condition, ValueOpcode::ConditionRef);
	if (term.kind != CFG::TerminatorKind::ConditionalBranch || term.true_block != t.header ||
	    t.vote == nullptr || t.vote->Arg(0) != alive ||
	    t.vote->Flags<CFG::BranchCondition>() != CFG::BranchCondition::ExecNonZero) {
		return Fail(__LINE__);
	}
	auto* voting = t.vote->Parent();
	if (voting != t.latch &&
	    (t.latch->ImmPredecessors().size() != 1 || t.latch->ImmPredecessors()[0] != voting)) {
		return Fail(__LINE__);
	}
	const auto* moved = Back(t, *t.extent).Resolve().TryInstruction();
	for (auto it = Position(*t.vote); it != voting->end(); ++it) {
		if (&*it == moved) return Fail(__LINE__);
	}
	if (voting != t.latch && moved != nullptr && moved->Parent() == t.latch) return Fail(__LINE__);

	const auto exec   = Value(t.exec);
	const auto inside = [&](const Block* block) { return Inside(t, block); };
	for (const auto* block: t.loop) {
		if (block->terminator.kind == CFG::TerminatorKind::Return ||
		    (block != t.latch && !std::ranges::all_of(block->ImmSuccessors(), inside))) {
			return Fail(__LINE__);
		}
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (op == ValueOpcode::BvhIntersect && &inst != &bvh) return Fail(__LINE__);
			if (!HasSideEffects(op) || op == ValueOpcode::Reference) continue;
			uint32_t data = 3;
			if (op == ValueOpcode::WriteSharedU32) {
				data = 1;
			} else if (op != ValueOpcode::StoreAddressU32) {
				return Fail(__LINE__);
			}
			const auto* pushed = Producer(ResolveActiveU32(inst.Arg(data), exec),
			                              ValueOpcode::CompositeExtractU32x4);
			if (pushed == nullptr || pushed->Arg(0) != Value(&bvh)) return Fail(__LINE__);
		}
	}
	return true;
}

// A walk over 64-bit node pointers: one loop takes the TLAS and, from instance records, the BLASes.
bool Match(const Program& program, Inst& bvh, Traversal& t) {
	const auto* descriptor = Producer(bvh.Arg(0), ValueOpcode::CompositeConstructU32x4);
	const auto* address    = Producer(bvh.Arg(1), ValueOpcode::MakeImageAddress);
	if (descriptor == nullptr || address == nullptr || bvh.Flags<uint32_t>() != 2u ||
	    !MatchLoop(program, bvh, address->Arg(2), t)) {
		return Fail(__LINE__);
	}
	for (uint32_t word = 0; word < 4; ++word) {
		if (Immediate(descriptor->Arg(word)) != TraversalDescriptor[word]) return Fail(__LINE__);
	}

	// The ray: extent, origin and direction phis start as the world ray and its t_max.
	for (uint32_t arg = 2; arg < 9; ++arg) {
		const auto* ray = HeaderPhi(t, address->Arg(arg));
		if (ray == nullptr) return Fail(__LINE__);
		t.ray[(arg + 4) % 7] = Entry(t, *ray);
	}

	// The node address is node + BVH header >> 3; the header starts as the TLAS, read from the
	// flattened SRT.
	const auto  exec  = Value(t.exec);
	const auto  lanes = Entry(t, *t.exec);
	const auto* sum   = Producer(ResolveActiveU32(address->Arg(0), exec), ValueOpcode::IAdd32);
	if (sum == nullptr) return Fail(__LINE__);
	auto* node      = HeaderPhi(t, sum->Arg(0));
	auto  base_part = sum->Arg(1);
	if (node == nullptr) {
		node      = HeaderPhi(t, sum->Arg(1));
		base_part = sum->Arg(0);
	}
	const auto* low = Producer(ResolveActiveU32(base_part, exec), ValueOpcode::CompositeExtractU64);
	const auto* shift =
	    low != nullptr ? Producer(low->Arg(0), ValueOpcode::ShiftRightLogical64) : nullptr;
	const auto* base =
	    shift != nullptr ? Producer(shift->Arg(0), ValueOpcode::CompositeConstructU64) : nullptr;
	if (node == nullptr || base == nullptr || Immediate(low->Arg(1)) != 0u ||
	    Immediate(shift->Arg(1)) != 3u) {
		return Fail(__LINE__);
	}
	for (uint32_t word = 0; word < 2; ++word) {
		const auto* header = HeaderPhi(t, base->Arg(word));
		const auto  slot =
		    header != nullptr ? Slot(ResolveActiveU32(Entry(t, *header), lanes)) : std::nullopt;
		if (!slot) return Fail(__LINE__);
		t.tlas[word] = *slot;
	}

	// The node takes the instance root link (+56) and prim record word 0; the record and the
	// instance are the addresses of those loads.
	std::unordered_set<const Inst*> nodes;
	Collect(Back(t, *node), t.header, nodes);
	const Inst* root  = nullptr;
	const Inst* word0 = nullptr;
	for (const auto* inst: nodes) {
		for (auto [load, offset]: {std::pair {&root, InstanceRoot}, std::pair {&word0, 0u}}) {
			if (!GlobalLoad(program, *inst, offset)) continue;
			if (*load != nullptr) return Fail(__LINE__);
			*load = inst;
		}
	}
	if (root == nullptr || word0 == nullptr) return Fail(__LINE__);

	// Prim record word 1 and the instance flags and BLAS are each loaded once in the loop.
	const std::array<std::pair<const Inst*, uint32_t>, 3> wanted {
	    {{word0, 4}, {root, InstanceFlags}, {root, InstanceBlas}}};
	std::array<const Inst*, 3> found {};
	for (const auto* block: t.loop) {
		for (const auto& inst: *block) {
			for (size_t load = 0; load < wanted.size(); ++load) {
				const auto& [like, offset] = wanted[load];
				if (!GlobalLoad(program, inst, offset) || inst.Arg(1) != like->Arg(1) ||
				    inst.Arg(2) != like->Arg(2)) {
					continue;
				}
				if (found[load] != nullptr) return Fail(__LINE__);
				found[load] = &inst;
			}
		}
	}
	const auto [word1, flags, blas] = found;
	if (word1 == nullptr || flags == nullptr || blas == nullptr) return Fail(__LINE__);
	t.loads = {word0, word1, flags};

	// The phi the iteration's code reads for each of them.
	std::vector<std::pair<Inst*, std::unordered_set<const Inst*>>> carried;
	for (auto& inst: *t.header) {
		if (inst.GetOpcode() == ValueOpcode::Phi && inst.GetType() == Type::U32 &&
		    DirectlyUsed(inst)) {
			carried.push_back({&inst, {}});
			Collect(Back(t, inst), t.header, carried.back().second);
		}
	}
	const std::array<const Inst*, 5> targets {word0, word1, root->Arg(1).Resolve().TryInstruction(),
	                                          root->Arg(2).Resolve().TryInstruction(), flags};
	t.state[0] = node;
	for (uint32_t role = 1; role < targets.size(); ++role) {
		for (auto& [carrier, values]: carried) {
			if (!values.contains(targets[role])) continue;
			if (t.state[role] != nullptr) return Fail(__LINE__);
			t.state[role] = carrier;
		}
		if (t.state[role] == nullptr ||
		    std::ranges::find(t.state.begin(), t.state.begin() + role, t.state[role]) !=
		        t.state.begin() + role) {
			return Fail(__LINE__);
		}
	}

	// The instance BLAS load, or a select that moves it into the next header, runs under the root
	// load's lanes and the mask test flags << k < 0: mask bit 23 - k.
	const auto leaf = root->Arg(3);
	const auto test = [&](Value condition) -> std::optional<uint32_t> {
		const auto* both = Producer(condition, ValueOpcode::LogicalAnd);
		if (both == nullptr || (both->Arg(0) != leaf && both->Arg(1) != leaf)) return std::nullopt;
		const auto* sign = Producer(both->Arg(0) == leaf ? both->Arg(1) : both->Arg(0),
		                            ValueOpcode::SGreaterThan32);
		const auto* mask = sign != nullptr ? Producer(ResolveActiveU32(sign->Arg(1), leaf),
		                                              ValueOpcode::ShiftLeftLogical32)
		                                   : nullptr;
		const auto  bit  = mask != nullptr ? Immediate(mask->Arg(1)) : std::nullopt;
		const bool  kept = bit && Immediate(sign->Arg(0)) == 0u && *bit >= 16u && *bit <= 23u &&
		                   ResolveActiveU32(mask->Arg(0), leaf).TryInstruction() == flags;
		return kept ? bit : std::nullopt;
	};
	auto                            bit = test(blas->Arg(3));
	std::unordered_set<const Inst*> next;
	if (!bit) Collect(Back(t, *HeaderPhi(t, base->Arg(0))), t.header, next);
	for (std::vector<const Inst*> taking {blas}; !bit && !taking.empty();) {
		const auto* value = taking.back();
		taking.pop_back();
		for (const auto& use: value->Uses()) {
			if (!bit && use.user->GetOpcode() == ValueOpcode::SelectU32 && use.operand == 1 &&
			    next.contains(use.user)) {
				bit = test(use.user->Arg(0));
				taking.push_back(use.user);
			}
		}
	}
	if (!bit) return Fail(__LINE__);
	t.cull_mask = 1u << (23u - *bit);
	return true;
}

// The 64-bit address whose bits 8..47 a BVH T#'s base holds, or empty; dword 3 must select
// barycentric triangle results.
Value DescriptorAddress(const Inst& descriptor) {
	const auto* low   = Producer(descriptor.Arg(0), ValueOpcode::CompositeExtractU64);
	const auto* field = low != nullptr ? Producer(low->Arg(0), ValueOpcode::BitwiseAnd64) : nullptr;
	const auto* shift =
	    field != nullptr ? Producer(field->Arg(0), ValueOpcode::ShiftRightLogical64) : nullptr;
	const auto* flags = Producer(descriptor.Arg(3), ValueOpcode::BitwiseOr32);
	const auto  bits  = flags != nullptr ? Immediate(flags->Arg(1)) : std::nullopt;
	if (shift == nullptr || Immediate(low->Arg(1)) != 0u || Immediate(shift->Arg(1)) != 8u ||
	    !bits || (*bits & Barycentrics) == 0u) {
		return {};
	}
	return shift->Arg(0).Resolve();
}

// The two dwords a T#'s address is built from directly, or nulls.
std::array<Inst*, 2> DescriptorWords(const Inst& descriptor) {
	const auto  address = DescriptorAddress(descriptor);
	const auto* words =
	    address.IsEmpty() ? nullptr : Producer(address, ValueOpcode::CompositeConstructU64);
	if (words == nullptr) return {};
	return {words->Arg(0).Resolve().TryInstruction(), words->Arg(1).Resolve().TryInstruction()};
}

// The address resource of `load` and the guest dword offset it reads from it, if it reads one.
std::optional<std::pair<const Inst*, uint32_t>> AddressOffset(const Program& program,
                                                              const Inst*    load) {
	if (load == nullptr || load->GetOpcode() != ValueOpcode::LoadAddressU32) return std::nullopt;
	const auto& memory   = program.memory_info[load->Flags<MemoryFlags>().index];
	const bool  based    = memory.kind == ResourceKind::ScalarAddress ||
	                       (memory.kind == ResourceKind::Global && !memory.address_is_full);
	const auto* resource = Producer(load->Arg(0), ValueOpcode::GetAddressResource);
	if (!based || resource == nullptr || Immediate(load->Arg(1)) != 0u) return std::nullopt;
	return std::pair {resource, memory.offset};
}

// The address resource of `load` if it reads the guest dword at `offset` from it, or nullptr.
const Inst* AddressWord(const Program& program, const Inst* load, uint32_t offset) {
	const auto at = AddressOffset(program, load);
	return at && at->second == offset ? at->first : nullptr;
}

// Whether two address resources take the same address.
bool SameAddress(const Inst& a, const Inst& b) {
	return a.Arg(0) == b.Arg(0) && a.Arg(1) == b.Arg(1);
}

// A BLAS address read from an instance record: the loads of its two dwords, the address resource
// of the low one and the layout.
struct Blas {
	std::array<Inst*, 2> loads {};
	const Inst*          instance = nullptr;
	BlasLayout           layout;
	Inst*                address = nullptr; // The 64-bit value of a node pointer's address.
};

// The BLAS of two dwords, if both load consecutive dwords of one instance.
std::optional<Blas> BlasOfWords(const Program& program, Value low, Value high, BlasLayout layout) {
	Blas blas;
	blas.loads        = {Producer(low, ValueOpcode::LoadAddressU32),
	                     Producer(high, ValueOpcode::LoadAddressU32)};
	const auto first  = AddressOffset(program, blas.loads[0]);
	const auto second = AddressOffset(program, blas.loads[1]);
	if (!first || !second || first->second % 4u != 0u || second->second != first->second + 4u ||
	    !SameAddress(*first->first, *second->first)) {
		return std::nullopt;
	}
	blas.instance      = first->first;
	blas.layout        = layout;
	blas.layout.offset = first->second;
	return blas;
}

// The BLAS of a 64-bit address: two instance dwords (low, high), or a node pointer
// (low, high & mask) << shift whose mask keeps the high dword's low bits under its flags.
std::optional<Blas> BlasOfAddress(const Program& program, Value address) {
	auto*      value = address.Resolve().TryInstruction();
	BlasLayout layout;
	if (value != nullptr && value->GetOpcode() == ValueOpcode::ShiftLeftLogical64) {
		const auto shift = Immediate(value->Arg(1));
		if (!shift || *shift == 0u || *shift > 8u) return std::nullopt;
		layout.shift = *shift;
	}
	const auto* words = layout.shift != 0u
	                        ? Producer(value->Arg(0), ValueOpcode::CompositeConstructU64)
	                        : Producer(address, ValueOpcode::CompositeConstructU64);
	if (words == nullptr) return std::nullopt;
	auto high = words->Arg(1);
	if (const auto* field = Producer(high, ValueOpcode::BitwiseAnd32); field != nullptr) {
		for (uint32_t side = 0; side < 2; ++side) {
			const auto mask = Immediate(field->Arg(side));
			if (!mask || *mask == 0u || ((*mask + 1u) & *mask) != 0u) continue;
			layout.mask = *mask;
			high        = field->Arg(1 - side);
			break;
		}
		if (layout.mask == ~0u) return std::nullopt;
	}
	if ((layout.shift != 0u) != (layout.mask != ~0u)) return std::nullopt;
	auto blas = BlasOfWords(program, words->Arg(0), high, layout);
	if (blas && layout.shift != 0u) blas->address = value;
	return blas;
}

// The BLAS of a 64-bit address's low and high halves: the halves of a node pointer's address, or
// the two instance dwords themselves.
std::optional<Blas> BlasOfHalves(const Program& program, const Inst* low, const Inst* high) {
	if (low == nullptr || high == nullptr) return std::nullopt;
	if (low->GetOpcode() == ValueOpcode::CompositeExtractU64 &&
	    high->GetOpcode() == ValueOpcode::CompositeExtractU64) {
		if (low->Arg(0) != high->Arg(0) || Immediate(low->Arg(1)) != 0u ||
		    Immediate(high->Arg(1)) != 1u) {
			return std::nullopt;
		}
		auto blas = BlasOfAddress(program, low->Arg(0));
		return blas && blas->layout.shift != 0u ? blas : std::nullopt;
	}
	return BlasOfWords(program, Value(const_cast<Inst*>(low)), Value(const_cast<Inst*>(high)),
	                   {});
}

bool SameBlas(const Blas& a, const Blas& b) {
	return a.loads == b.loads && a.layout == b.layout;
}

// The mask M of a test (field & M) != 0 of the instance mask in byte `byte` of the flags, or empty:
// the field is the flags shifted or extracted down to that byte, or the flags themselves with M
// within that byte. `flags` tells the flags word.
template <typename IsFlags>
Value ByteMask(Value test, uint32_t byte, IsFlags&& flags) {
	const auto* compare = Producer(test, ValueOpcode::INotEqual32);
	const Inst* bits    = nullptr;
	for (uint32_t side = 0; compare != nullptr && bits == nullptr && side < 2; ++side) {
		if (Immediate(compare->Arg(1 - side)) == 0u) {
			bits = Producer(compare->Arg(side), ValueOpcode::BitwiseAnd32);
		}
	}
	const auto is = [&](Value value) {
		const auto* inst = value.Resolve().TryInstruction();
		return inst != nullptr && flags(*inst);
	};
	for (uint32_t side = 0; bits != nullptr && side < 2; ++side) {
		const auto  mask  = bits->Arg(1 - side);
		const auto* field = bits->Arg(side).Resolve().TryInstruction();
		if (field == nullptr) continue;
		if (flags(*field)) {
			const auto immediate = Immediate(mask);
			if (immediate && *immediate != 0u && (*immediate & ~(0xffu << (8u * byte))) == 0u) {
				return Value(*immediate >> (8u * byte));
			}
			continue;
		}
		const auto op = field->GetOpcode();
		if (op == ValueOpcode::ShiftRightLogical32 && Immediate(field->Arg(1)) == 8u * byte &&
		    is(field->Arg(0))) {
			return mask;
		}
		if (op == ValueOpcode::BitFieldUExtract && Immediate(field->Arg(1)) == 8u * byte &&
		    Immediate(field->Arg(2)) == 8u && is(field->Arg(0))) {
			return mask;
		}
	}
	return {};
}

// The instance mask bits a branch on a test of the flags of `instance` checks, with the mask in
// byte `byte` of the flags.
std::optional<uint32_t> MaskTest(const Program& program, const Inst& branch, const Inst& instance,
                                 uint32_t byte) {
	const auto mask = ByteMask(branch.Arg(0), byte, [&](const Inst& inst) {
		const auto* at = AddressWord(program, &inst, InstanceFlags);
		return at != nullptr && SameAddress(*at, instance);
	});
	const auto bits = mask.IsEmpty() ? std::nullopt : Immediate(mask);
	if (!bits || *bits == 0u || *bits > 0xffu) return std::nullopt;
	return *bits;
}

// Whether the low instance address word, a sum, is the TLAS header's instance array, read from the
// flattened SRT `slot`, plus (node >> 3) << 7.
bool InstanceArray(const Inst& low, uint32_t slot) {
	for (uint32_t arg = 0; arg < 2; ++arg) {
		const auto* part = Producer(low.Arg(arg), ValueOpcode::CompositeExtractU64);
		const auto* scale =
		    part != nullptr ? Producer(part->Arg(0), ValueOpcode::ShiftLeftLogical64) : nullptr;
		const auto* index = scale != nullptr
		                        ? Producer(scale->Arg(0), ValueOpcode::CompositeConstructU64)
		                        : nullptr;
		const auto* node =
		    index != nullptr ? Producer(index->Arg(0), ValueOpcode::ShiftRightLogical32) : nullptr;
		const auto* array = Producer(low.Arg(1 - arg), ValueOpcode::IAdd32);
		if (node != nullptr && array != nullptr && Immediate(part->Arg(1)) == 0u &&
		    Immediate(scale->Arg(1)) == 7u && Immediate(index->Arg(1)) == 0u &&
		    Immediate(node->Arg(1)) == 3u &&
		    (Slot(array->Arg(0)) == slot || Slot(array->Arg(1)) == slot)) {
			return true;
		}
	}
	return Fail(__LINE__);
}

// The BLAS address words 0 and 1 of a V# over the BLAS's prim records (stride 8), or nulls.
std::array<const Inst*, 2> RecordBase(Value resource) {
	const auto* buffer = Producer(resource, ValueOpcode::GetBufferResource);
	const auto* stride =
	    buffer != nullptr ? Producer(buffer->Arg(1), ValueOpcode::BitwiseOr32) : nullptr;
	const auto bits = stride != nullptr ? Immediate(stride->Arg(1)) : std::nullopt;
	if (!bits || ((*bits >> 16) & 0x3fffu) != 8u) return {};
	return {buffer->Arg(0).Resolve().TryInstruction(), stride->Arg(0).Resolve().TryInstruction()};
}

// Whether a header phi's value is never read in the loop: it is only kept, by a select on the
// loop's lanes, for lanes the loop skips, or carried by phis, in the loop back to its header or out
// of it.
bool Kept(const Loop& t, const Inst& phi, const Inst& exec, std::unordered_set<const Inst*>& seen) {
	if (!seen.insert(&phi).second) return true;
	return std::ranges::all_of(phi.Uses(), [&](const Use& use) {
		const auto* user = use.user;
		if (user->GetOpcode() == ValueOpcode::SelectU32 && use.operand == 2 &&
		    user->Arg(0).Resolve().TryInstruction() == &exec) {
			return true;
		}
		return user->GetOpcode() == ValueOpcode::Phi &&
		       (!Inside(t, user->Parent()) || Kept(t, *user, exec, seen));
	});
}

bool Kept(const Loop& t, const Inst& phi, const Inst& exec) {
	std::unordered_set<const Inst*> seen;
	return Kept(t, phi, exec, seen);
}

// Whether `user` in the loop body takes the candidate's instance address and BLAS words in place
// of the walk's: all but the intersection's T# and address and the prim record V# do.
bool Replaced(const Traversal& t, const Inst& user) {
	const auto other = [&](Value value) { return value.Resolve().TryInstruction() != &user; };
	return user.Parent() != t.header && Inside(t, user.Parent()) && other(t.bvh->Arg(0)) &&
	       other(t.bvh->Arg(1)) && other(t.record->Arg(0));
}

// A BLAS walk over 32-bit node pointers inside a TLAS walk. The TLAS walk intersects its nodes with
// the world ray under a T# over the TLAS header the flattened SRT holds; its instance leaf node
// loads instance record node >> 3 of the header's instance array and enters a loop under a T# over
// the BLAS at the record's +0. That loop reads prim records through a V# over the BLAS with stride
// 8: word 0 is the next node and word 1 a value the loop keeps.
bool MatchNested(const Program& program, Inst& bvh, std::span<Inst* const> intersections,
                 Traversal& t) {
	const auto* descriptor = Producer(bvh.Arg(0), ValueOpcode::CompositeConstructU32x4);
	const auto* address    = Producer(bvh.Arg(1), ValueOpcode::MakeImageAddress);
	if (descriptor == nullptr || address == nullptr ||
	    !MatchLoop(program, bvh, address->Arg(1), t)) {
		return Fail(__LINE__);
	}
	auto*       node     = HeaderPhi(t, address->Arg(0));
	const auto  target   = DescriptorAddress(*descriptor);
	const auto  blas     = target.IsEmpty() ? std::nullopt : BlasOfAddress(program, target);
	auto*       word0    = blas ? blas->loads[0] : nullptr;
	auto*       word1    = blas ? blas->loads[1] : nullptr;
	const auto* instance = blas ? blas->instance : nullptr;
	if (node == nullptr || instance == nullptr || Inside(t, word0->Parent()) ||
	    Inside(t, word1->Parent())) {
		return Fail(__LINE__);
	}
	auto* low  = Producer(instance->Arg(0), ValueOpcode::IAdd32);
	auto* high = instance->Arg(1).Resolve().TryInstruction();
	if (low == nullptr || high == nullptr) return Fail(__LINE__);
	t.instance = {low, high, word0, word1};
	t.blas     = blas->layout;
	t.swaps    = {{low, 0u}, {high, 1u}};
	if (blas->address == nullptr) {
		// The BLAS address is the two dwords.
		t.swaps.insert(t.swaps.end(), {{word0, 2u}, {word1, 3u}});
	} else {
		// The halves of a node pointer's address, and its dwords, which also hold flags.
		for (const auto& use: blas->address->Uses()) {
			const auto half = Immediate(use.user->Arg(1));
			if (use.user->GetOpcode() == ValueOpcode::CompositeExtractU64 && half && *half < 2u) {
				t.swaps.emplace_back(use.user, 2u + *half);
			}
		}
		t.swaps.insert(t.swaps.end(), {{word0, 4u}, {word1, 5u}});
	}

	// The TLAS walk's intersection, in a loop around this one.
	const Inst* walk = nullptr;
	for (const auto* other: intersections) {
		const auto* around = EnclosingLoop(program, other->Parent());
		if (other->Flags<uint32_t>() != 1u || around == nullptr ||
		    BlockIndex(program, around) >= BlockIndex(program, t.header) ||
		    BlockIndex(program, around->terminator.continue_block) < BlockIndex(program, t.latch)) {
			continue;
		}
		const auto* words     = Producer(other->Arg(0), ValueOpcode::CompositeConstructU32x4);
		const auto  tlas      = words != nullptr ? DescriptorWords(*words) : std::array<Inst*, 2> {};
		const auto  low_slot  = tlas[0] != nullptr ? Slot(Value(tlas[0])) : std::nullopt;
		const auto  high_slot = tlas[1] != nullptr ? Slot(Value(tlas[1])) : std::nullopt;
		if (!low_slot || !high_slot) continue;
		if (walk != nullptr) return Fail(__LINE__);
		walk   = other;
		t.tlas = {*low_slot, *high_slot};
	}
	if (walk == nullptr) return Fail(__LINE__);

	// Its world ray is defined before the outermost loop around it; t_max is this loop's extent.
	const auto* ray   = Producer(walk->Arg(1), ValueOpcode::MakeImageAddress);
	auto*       outer = EnclosingLoop(program, walk->Parent());
	while (auto* up = EnclosingLoop(program, outer)) {
		outer = up;
	}
	if (ray == nullptr) return Fail(__LINE__);
	for (uint32_t arg = 2; arg < 8; ++arg) {
		const auto* value = ray->Arg(arg).Resolve().TryInstruction();
		if (value != nullptr &&
		    BlockIndex(program, value->Parent()) >= BlockIndex(program, outer)) {
			return Fail(__LINE__);
		}
		t.ray[arg - 2] = ray->Arg(arg);
	}
	t.ray[6] = Entry(t, *t.extent);

	if (!InstanceArray(*low, t.tlas[0])) return Fail(__LINE__);

	// The prim record load whose word 0 is the next node, and the phi that keeps word 1.
	std::unordered_set<const Inst*> next;
	Collect(Back(t, *node), t.header, next);
	Inst* record = nullptr;
	for (const auto* inst: next) {
		auto* load =
		    inst->GetOpcode() == ValueOpcode::CompositeExtractU32x2 && Immediate(inst->Arg(1)) == 0u
		        ? Producer(inst->Arg(0), ValueOpcode::LoadBufferU32x2)
		        : nullptr;
		if (load == nullptr) continue;
		if (record != nullptr && record != load) return Fail(__LINE__);
		record = load;
	}
	t.record                 = record;
	const auto record_base   = record != nullptr ? RecordBase(record->Arg(0))
	                                             : std::array<const Inst*, 2> {};
	const auto record_blas   = BlasOfHalves(program, record_base[0], record_base[1]);
	if (record == nullptr || !record_blas || !SameBlas(*record_blas, *blas) ||
	    Immediate(record->Arg(2)) != 0u || Immediate(record->Arg(3)) != 0u ||
	    !Inside(t, record->Parent())) {
		return Fail(__LINE__);
	}
	const auto& memory = program.memory_info[record->Flags<MemoryFlags>().index];
	if (!memory.idxen || memory.offen || memory.offset != 0u) return Fail(__LINE__);
	const auto record_uses = record->Uses();
	const auto second      = std::ranges::find_if(record_uses, [](const Use& use) {
		return use.user->GetOpcode() == ValueOpcode::CompositeExtractU32x2 &&
		       Immediate(use.user->Arg(1)) == 1u;
	});
	if (second == record_uses.end()) return Fail(__LINE__);
	for (auto& inst: *t.header) {
		std::unordered_set<const Inst*> values;
		if (inst.GetOpcode() == ValueOpcode::Phi && inst.GetType() == Type::U32 && &inst != node &&
		    DirectlyUsed(inst)) {
			Collect(Back(t, inst), t.header, values);
		}
		if (!values.contains(second->user)) continue;
		if (t.state[1] != nullptr) return Fail(__LINE__);
		t.state[1] = &inst;
	}
	if (t.state[1] == nullptr) return Fail(__LINE__);
	t.state[0] = node;

	// Of what the loop reads of the instance, the candidate's address and BLAS words replace its
	// own; the rest must be the intersection's T# and ray, the record V#, or the start of a header
	// phi the query overrides or that only lanes the loop skips keep. A branch between the
	// instance and the loop may test the instance mask, which the query's cull mask applies.
	const auto                            first = BlockIndex(program, instance->Parent());
	const auto                            begin = BlockIndex(program, t.header);
	const auto                            last  = BlockIndex(program, t.latch);
	std::optional<uint32_t>               mask;
	std::unordered_set<const Inst*>       seen;
	std::vector<std::pair<Inst*, size_t>> reads;
	for (std::vector<const Inst*> work {low, high}; !work.empty();) {
		const auto* value = work.back();
		work.pop_back();
		if (!seen.insert(value).second) continue;
		const bool swapped = std::ranges::any_of(
		    t.swaps, [&](const auto& swap) { return swap.first == value; });
		for (const auto& use: value->Uses()) {
			auto*      user = use.user;
			const auto at   = BlockIndex(program, user->Parent());
			if (at < first || at > last) continue;
			if (at < begin && user->GetOpcode() != ValueOpcode::ConditionRef) {
				work.push_back(user);
			} else if (at < begin) {
				if (mask) return Fail(__LINE__);
				mask = MaskTest(program, *user, *instance, t.blas.MaskByte());
				if (!mask) return Fail(__LINE__);
			} else if (user->Parent() == t.header) {
				if (user != node && !Kept(t, *user, *t.exec)) return Fail(__LINE__);
			} else if (Replaced(t, *user)) {
				if (!swapped) return Fail(__LINE__);
				reads.emplace_back(user, use.operand);
			}
		}
	}
	t.cull_mask = mask.value_or(0xffu);

	// Per lane, the candidate's instance and BLAS reach the loop's code that read the walk's. Its
	// scalar loads become per-lane loads; whole-wave operations take per-lane values, but V#s, T#s,
	// S#s, constant buffer reads and lane numbers must stay uniform.
	while (!reads.empty()) {
		const auto [inst, operand] = reads.back();
		reads.pop_back();
		const auto op = inst->GetOpcode();
		if (!Inside(t, inst->Parent())) continue;
		if (op == ValueOpcode::ReadConstBuffer || op == ValueOpcode::GetBufferResource ||
		    op == ValueOpcode::GetImageResource || op == ValueOpcode::GetSamplerResource ||
		    ((op == ValueOpcode::ReadLane || op == ValueOpcode::WriteLane) && operand != 0)) {
			return Fail(__LINE__);
		}
		if (op == ValueOpcode::Ballot || op == ValueOpcode::ReadFirstLane ||
		    op == ValueOpcode::ReadLane || op == ValueOpcode::WriteLane ||
		    op == ValueOpcode::SubgroupReduceU32 || !seen.insert(inst).second) {
			continue;
		}
		if (op == ValueOpcode::LoadAddressU32 &&
		    program.memory_info[inst->Flags<MemoryFlags>().index].kind ==
		        ResourceKind::ScalarAddress) {
			t.scalar.push_back(inst);
		}
		for (const auto& use: inst->Uses()) {
			reads.emplace_back(use.user, use.operand);
		}
	}
	return true;
}

// A walk that takes TLAS and BLAS nodes off one stack its wave shares: a loop around a BLAS loop
// around a loop over a leaf's prim records, whose triangle tests commit the first hit a lane
// accepts, which then leaves.
struct Walk: Loop {
	Inst*                skip  = nullptr; // The vote that enters the walk's body.
	Inst*                leave = nullptr; // The vote that leaves the walk.
	Value                lanes;
	std::array<Value, 7> ray {}; // World origin xyz, direction xyz, t_max.
	Inst*                bvh   = nullptr;
	Inst*                flags = nullptr;         // The instance flags load.
	Inst*                word1 = nullptr;         // The prim record loop's phi of record word 1.
	std::array<Inst*, 2> instance {};             // The instance address.
	std::array<Inst*, 3> lanes_in {};             // What the walk's body, its record loop and that
	                                              // loop's entry run on.
	Inst*                accepted = nullptr;      // Whether a lane tests the instance's records.
	Inst*                commit   = nullptr;      // Whether a lane commits the hit it tests.
	std::vector<std::pair<Inst*, Value>> commits; // The walk's exit phi and its committed value.
	std::array<uint32_t, 2>              tlas {};
	std::optional<BlasLayout>            blas;
	// The walk's other loads of instance dwords, by offset.
	std::unordered_map<const Inst*, uint32_t> words;
};

// The values the per-candidate code of a walk takes from the candidate.
enum Leaf { Lanes, Numerator, Denominator, Flags, Word1, InstanceLow, InstanceHigh, None };

Leaf LeafOf(const Walk& w, const Inst& inst) {
	if (std::ranges::find(w.lanes_in, &inst) != w.lanes_in.end()) return Lanes;
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU32x4 && inst.Arg(0) == Value(w.bvh)) {
		const auto index = Immediate(inst.Arg(1));
		if (index == 0u) return Numerator;
		return index == 1u ? Denominator : None;
	}
	if (&inst == w.flags) return Flags;
	if (&inst == w.word1) return Word1;
	if (&inst == w.instance[0]) return InstanceLow;
	return &inst == w.instance[1] ? InstanceHigh : None;
}

// A select that keeps a value for lanes the walk does not run: on a candidate, its first arm.
bool LaneSelect(const Walk& w, const Inst& inst) {
	const auto* lanes = inst.Arg(0).Resolve().TryInstruction();
	return (inst.GetOpcode() == ValueOpcode::SelectU32 ||
	        inst.GetOpcode() == ValueOpcode::SelectU1) &&
	       lanes != nullptr && LeafOf(w, *lanes) == Lanes;
}

// Whether an operation computes a lane's value from that lane's arguments alone, not from wave
// state, other lanes or memory.
bool PerLane(ValueOpcode op) {
	using Op = ValueOpcode;
	constexpr std::array excluded {
	    Op::ConditionRef,      Op::ReadFirstLane, Op::ReadLane,     Op::WriteLane,
	    Op::SubgroupReduceU32, Op::Permlane16U32, Op::PermuteU32,   Op::BpermuteU32,
	    Op::SwizzleU32,        Op::DppMoveU32,    Op::DppUpdateU32, Op::LoadDeviceAddressU32};
	const auto plain = [](Type type) {
		return type == Type::U1 || type == Type::U32 || type == Type::F32;
	};
	if (!plain(TypeOf(op)) || NumArgsOf(op) == 0 || HasSideEffects(op) ||
	    SharedAccessOf(op) != SharedAccess::None ||
	    std::ranges::find(excluded, op) != excluded.end()) {
		return Fail(__LINE__);
	}
	for (size_t arg = 0; arg < NumArgsOf(op); ++arg) {
		if (!plain(ArgTypeOf(op, arg))) return Fail(__LINE__);
	}
	return true;
}

// What the walk's per-candidate code computes for `value`, copied to `at` from the candidate's
// `natives`, or without `at` `value` itself; empty unless it depends, past the leaves, only on
// values from before the walk.
// `word` loads the candidate's instance dword at an offset.
Value Copy(const Walk& w, Value value, std::unordered_map<const Inst*, Value>& copies,
           const Inserter* at = nullptr, std::span<const Value> natives = {},
           const std::function<Value(uint32_t)>& word = {}) {
	value            = value.Resolve();
	auto* const inst = value.TryInstruction();
	if (inst == nullptr || !Inside(w, inst->Parent())) return value;
	if (const auto leaf = LeafOf(w, *inst); leaf != None) {
		return at != nullptr ? natives[leaf] : value;
	}
	if (const auto found = copies.find(inst); found != copies.end()) return found->second;
	Value copy;
	if (const auto offset = w.words.find(inst); offset != w.words.end()) {
		copy = at != nullptr ? word(offset->second) : value;
	} else if (LaneSelect(w, *inst)) {
		copy = Copy(w, inst->Arg(1), copies, at, natives, word);
	} else if (PerLane(inst->GetOpcode())) {
		std::vector<Value> args;
		for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
			args.push_back(Copy(w, inst->Arg(arg), copies, at, natives, word));
		}
		if (std::ranges::none_of(args, &Value::IsEmpty)) {
			copy = at != nullptr ? (*at)(*inst, args) : value;
		}
	}
	copies.emplace(inst, copy);
	return copy;
}

// The mask M of a conjunct (field & M) != 0 of `test` over the instance mask's byte of the flags,
// M from before the walk, or 0xff.
Value CullMask(const Walk& w, Value test) {
	const auto* inst = test.Resolve().TryInstruction();
	if (inst != nullptr && inst->GetOpcode() == ValueOpcode::LogicalAnd) {
		for (uint32_t arg = 0; arg < 2; ++arg) {
			if (const auto mask = CullMask(w, inst->Arg(arg)); mask != Value(0xffu)) return mask;
		}
		return Value(0xffu);
	}
	const auto byte = w.blas.value_or(BlasLayout {}).MaskByte();
	const auto mask =
	    inst != nullptr ? ByteMask(test, byte, [&](const Inst& flags) { return &flags == w.flags; })
	                    : Value {};
	const auto* defined = mask.IsEmpty() ? nullptr : mask.Resolve().TryInstruction();
	if (mask.IsEmpty() || (defined != nullptr && Inside(w, defined->Parent()))) return Value(0xffu);
	return mask;
}

bool MatchWalk(const Program& program, Inst& bvh, std::span<Inst* const> intersections, Walk& w) {
	Loop records, blases;
	if (!Enter(program, EnclosingLoop(program, bvh.Parent()), records) ||
	    !Enter(program, EnclosingLoop(program, records.header), blases) ||
	    !Enter(program, EnclosingLoop(program, blases.header), w) ||
	    w.header->terminator.kind != CFG::TerminatorKind::Branch ||
	    w.header->terminator.true_block == nullptr || w.latch->ImmPredecessors().size() != 1) {
		return Fail(__LINE__);
	}
	// The body runs from a vote after the header; a vote before the latch leaves.
	auto* enter = w.header->terminator.true_block;
	auto* vote  = w.latch->ImmPredecessors()[0];
	w.skip      = Producer(enter->condition, ValueOpcode::ConditionRef);
	w.leave     = Producer(vote->condition, ValueOpcode::ConditionRef);
	if (w.skip == nullptr || w.leave == nullptr ||
	    enter->terminator.kind != CFG::TerminatorKind::ConditionalBranch ||
	    enter->terminator.false_block != vote ||
	    vote->terminator.kind != CFG::TerminatorKind::ConditionalBranch ||
	    vote->terminator.false_block != w.latch || Inside(w, vote->terminator.true_block)) {
		return Fail(__LINE__);
	}
	const auto inside = [&](const Block* block) { return Inside(w, block); };
	for (const auto* block: w.loop) {
		if (block != vote && !std::ranges::all_of(block->ImmSuccessors(), inside)) return Fail(__LINE__);
		for (const auto& inst: *block) {
			if (HasSideEffects(inst.GetOpcode()) && inst.GetOpcode() != ValueOpcode::Reference) {
				return Fail(__LINE__);
			}
		}
	}
	const auto* body = Producer(w.skip->Arg(0), ValueOpcode::LogicalAnd);
	for (uint32_t arg = 0; body != nullptr && arg < 2; ++arg) {
		const auto* lanes = body->Arg(arg).Resolve().TryInstruction();
		if (lanes != nullptr && !Inside(w, lanes->Parent())) w.lanes = body->Arg(arg);
	}
	auto* exec = HeaderPhi(records, bvh.Arg(2));
	if (w.lanes.IsEmpty() || exec == nullptr) return Fail(__LINE__);
	w.bvh      = &bvh;
	w.lanes_in = {w.skip->Arg(0).Resolve().TryInstruction(), exec,
	              Entry(records, *exec).Resolve().TryInstruction()};

	// The TLAS walk's intersection, in the walk's own loop, takes the world ray.
	uint32_t found = 0;
	for (const auto* other: intersections) {
		if (other->Flags<uint32_t>() != 1u || EnclosingLoop(program, other->Parent()) != w.header) {
			continue;
		}
		const auto* words = Producer(other->Arg(0), ValueOpcode::CompositeConstructU32x4);
		const auto  tlas  = words != nullptr ? DescriptorWords(*words) : std::array<Inst*, 2> {};
		const auto* ray   = Producer(other->Arg(1), ValueOpcode::MakeImageAddress);
		const auto  low   = tlas[0] != nullptr ? Slot(Value(tlas[0])) : std::nullopt;
		const auto  high  = tlas[1] != nullptr ? Slot(Value(tlas[1])) : std::nullopt;
		if (!low || !high || ray == nullptr || ++found > 1) return Fail(__LINE__);
		w.tlas = {*low, *high};
		for (uint32_t arg = 1; arg < 8; ++arg) {
			const auto* value = ray->Arg(arg).Resolve().TryInstruction();
			if (value != nullptr &&
			    BlockIndex(program, value->Parent()) >= BlockIndex(program, w.header)) {
				return Fail(__LINE__);
			}
			w.ray[(arg + 5) % 7] = ray->Arg(arg);
		}
	}
	if (found == 0) return Fail(__LINE__);

	// The record loop's commit: a branch on whether a lane commits into selects of the committed
	// values over record loop phis, which the BLAS loop's phis start from the walk's header phis;
	// exit phis take those past a skipped body.
	for (auto* block: records.loop) {
		const auto* ref = Producer(block->condition, ValueOpcode::ConditionRef);
		if (ref == nullptr || block->terminator.kind != CFG::TerminatorKind::ConditionalBranch) {
			continue;
		}
		// Each select's old record loop phi and new value.
		std::vector<std::pair<const Inst*, Value>> selects;
		bool                                       kept = true;
		for (auto& inst: *block->terminator.true_block) {
			if (inst.GetOpcode() != ValueOpcode::SelectU32 || inst.Arg(0) != ref->Arg(0)) continue;
			const auto* old = HeaderPhi(records, inst.Arg(2));
			kept            = kept && old != nullptr;
			selects.emplace_back(old, inst.Arg(1));
		}
		if (selects.empty() || !kept) continue;
		if (w.commit != nullptr) return Fail(__LINE__);
		w.commit = ref->Arg(0).Resolve().TryInstruction();
		for (const auto& [old, committed]: selects) {
			const auto* outer = HeaderPhi(blases, Entry(records, *old));
			auto*       value = outer != nullptr ? HeaderPhi(w, Entry(blases, *outer)) : nullptr;
			if (value == nullptr) return Fail(__LINE__);
			Inst* exit = nullptr;
			for (auto& inst: *vote) {
				if (inst.GetOpcode() != ValueOpcode::Phi ||
				    inst.Arg(Incoming(inst, enter)) != Value(value)) {
					continue;
				}
				if (exit != nullptr) return Fail(__LINE__);
				exit = &inst;
			}
			if (exit == nullptr) return Fail(__LINE__);
			w.commits.emplace_back(exit, committed);
		}
	}
	if (w.commit == nullptr) return Fail(__LINE__);

	// The instance flags load reads the instance the TLAS leaf names, and its block's branch into
	// the BLAS loop is whether a lane tests the instance's records.
	for (auto* block: w.loop) {
		for (auto& inst: *block) {
			if (AddressWord(program, &inst, InstanceFlags) == nullptr) continue;
			if (w.flags != nullptr) return Fail(__LINE__);
			w.flags = &inst;
		}
	}
	const auto* address = AddressWord(program, w.flags, InstanceFlags);
	const auto* ref     = address != nullptr
	                          ? Producer(w.flags->Parent()->condition, ValueOpcode::ConditionRef)
	                          : nullptr;
	if (ref == nullptr || w.flags->Parent()->terminator.true_block != blases.preheader) {
		return Fail(__LINE__);
	}
	w.instance = {Producer(address->Arg(0), ValueOpcode::IAdd32),
	              address->Arg(1).Resolve().TryInstruction()};
	w.accepted = ref->Arg(0).Resolve().TryInstruction();
	if (w.instance[0] == nullptr || w.instance[1] == nullptr || w.accepted == nullptr ||
	    !InstanceArray(*w.instance[0], w.tlas[0])) {
		return Fail(__LINE__);
	}

	// Record word 1 is read at +4 of each record through a V# over the BLAS the instance gives.
	const auto blas = [&](const std::array<const Inst*, 2>& base) -> std::optional<BlasLayout> {
		const auto found = BlasOfHalves(program, base[0], base[1]);
		if (!found || !SameAddress(*found->instance, *address)) return std::nullopt;
		return found->layout;
	};
	for (auto& phi: *records.header) {
		if (phi.GetOpcode() != ValueOpcode::Phi) continue;
		bool                      word1 = phi.NumArgs() != 0;
		std::optional<BlasLayout> layout;
		for (size_t arg = 0; word1 && arg < phi.NumArgs(); ++arg) {
			const auto* read = Producer(phi.Arg(arg), ValueOpcode::ReadConstBuffer);
			const auto  base =
			    read != nullptr ? RecordBase(read->Arg(0)) : std::array<const Inst*, 2> {};
			const auto  each = blas(base);
			word1 = each && (!layout || *layout == *each) &&
			        program.memory_info[read->Flags<MemoryFlags>().index].offset == 4u;
			layout = each;
		}
		if (!word1) continue;
		if (w.word1 != nullptr) return Fail(__LINE__);
		w.word1 = &phi;
		w.blas  = layout;
	}
	if (w.word1 == nullptr) return Fail(__LINE__);

	// The candidate's instance gives the walk's other loads of instance dwords.
	for (auto* block: w.loop) {
		for (auto& inst: *block) {
			const auto at = AddressOffset(program, &inst);
			if (&inst != w.flags && at && SameAddress(*at->first, *address)) {
				w.words.emplace(&inst, at->second);
			}
		}
	}

	std::unordered_map<const Inst*, Value> copies;
	const auto copied = [&](Value value) { return !Copy(w, value, copies).IsEmpty(); };
	return copied(Value(w.accepted)) && copied(Value(w.commit)) &&
	       std::ranges::all_of(w.commits,
	                           [&](const auto& commit) { return copied(commit.second); });
}

// The BLAS address of an instance's dwords `low` and `high` in `layout`.
Value BlasAddress(const Inserter& at, Value low, Value high, const BlasLayout& layout) {
	using Op = ValueOpcode;
	if (layout.mask != ~0u) high = at(Op::BitwiseAnd32, {high, Value(layout.mask)});
	auto address = at(Op::CompositeConstructU64, {low, high});
	if (layout.shift != 0u) address = at(Op::ShiftLeftLogical64, {address, Value(layout.shift)});
	return address;
}

void Rewrite(Program& program, Traversal& t) {
	using Op = ValueOpcode;
	// The preheader starts the query on the loop's lanes, which then run while it has candidates.
	const Inserter entry {t.preheader, t.preheader->end()};
	const auto     lanes   = Entry(t, *t.exec);
	const auto     valid   = entry(Op::RayQueryValid, {});
	const auto     invalid = entry(Op::LogicalNot, {valid});
	const auto     start   = entry(Op::LogicalAnd, {valid, lanes});
	entry(Op::RayQueryInitialize, {t.ray[0], t.ray[1], t.ray[2], t.ray[3], t.ray[4], t.ray[5],
	                               t.ray[6], Value(t.cull_mask), start});
	const auto entering = entry(Op::SelectU1, {valid, entry(Op::RayQueryProceed, {start}), lanes});
	for (auto* lane: t.lanes) {
		lane->SetArg(Incoming(*lane, t.preheader), entering);
	}

	// The body reads the candidate's prim record and instance where the loop reads its own.
	auto*          body = t.header->terminator.true_block;
	const Inserter at {body, std::ranges::find_if(*body, [](const Inst& inst) {
		                   return inst.GetOpcode() != Op::Phi;
	                   })};
	const auto     current   = at(Op::LogicalAnd, {valid, Value(t.exec)});
	const auto     candidate = at(Op::RayQueryCandidate, {current});
	const auto     instance  = at(Op::RayQueryCandidateInstance, {current});
	const auto     part      = [&](Op op, Value value, uint32_t index) {
		return at(op, {value, Value(index)});
	};
	const auto instance_low  = part(Op::CompositeExtractU64, instance, 0);
	const auto instance_high = part(Op::CompositeExtractU64, instance, 1);
	const auto offset        = at(
	    Op::ShiftLeftLogical64,
	    {at(Op::CompositeConstructU64, {part(Op::CompositeExtractU32x4, candidate, 3), Value(0u)}),
		 Value(3u)});
	const auto choose = [&](Inst& phi, Value native) {
		const auto chosen = at(Op::SelectU32, {valid, native, Value(&phi)});
		Redirect(phi, chosen, *chosen.Instruction());
	};
	// Per-lane loads of guest dwords `word` 0 and 1 at `base` from an address: a flat walk's like its
	// prim record loads, a nested walk's under new full-address memory infos.
	const auto info = [&](uint32_t base, uint32_t word) {
		if (t.instance[0] == nullptr) return t.loads[word]->Flags<uint64_t>();
		const MemoryFlags flags {.index = static_cast<uint32_t>(program.memory_info.size()),
		                         .pc    = t.instance[2]->Flags<MemoryFlags>().pc};
		program.memory_info.push_back(
		    {.kind = ResourceKind::Global, .offset = base + word * 4u, .address_is_full = true});
		return std::bit_cast<uint64_t>(flags);
	};
	const std::array words {info(0, 0), info(0, 1)};
	const std::array blas_words {info(t.blas.offset, 0), info(t.blas.offset, 1)};
	const auto       load = [&](Value low, Value high, uint64_t flags) {
		return at(Op::LoadAddressU32, {at(Op::GetAddressResource, {low, high}), low, high, current},
		          flags);
	};
	// The BLAS header is built from two instance dwords.
	const std::array raw {load(instance_low, instance_high, blas_words[0]),
	                      load(instance_low, instance_high, blas_words[1])};
	const auto       blas_address = BlasAddress(at, raw[0], raw[1], t.blas);
	const auto record = at(Op::IAdd64, {blas_address, offset});
	const auto low    = part(Op::CompositeExtractU64, record, 0);
	const auto high   = part(Op::CompositeExtractU64, record, 1);
	choose(*t.state[0], load(low, high, words[0]));
	choose(*t.state[1], load(low, high, words[1]));
	if (t.instance[0] == nullptr) {
		choose(*t.state[2], instance_low);
		choose(*t.state[3], instance_high);
		choose(*t.state[4], load(instance_low, instance_high, t.loads[2]->Flags<uint64_t>()));
	} else {
		// The loop's own instance address and BLAS words become the candidate's, which its scalar
		// loads read per lane. The record load only runs without the query.
		const std::array native {instance_low,
		                         instance_high,
		                         part(Op::CompositeExtractU64, blas_address, 0),
		                         part(Op::CompositeExtractU64, blas_address, 1),
		                         raw[0],
		                         raw[1]};
		for (const auto& [swapped, index]: t.swaps) {
			auto&      value  = *swapped;
			const auto chosen = at(Op::SelectU32, {valid, native[index], Value(&value)});
			const auto uses   = value.Uses();
			for (const auto& use: uses) {
				if (use.user != chosen.Instruction() && Replaced(t, *use.user)) {
					use.user->SetArg(use.operand, chosen);
				}
			}
		}
		for (auto* scalar: t.scalar) {
			// Each loads its one dword of the group.
			auto& memory           = program.memory_info[scalar->Flags<MemoryFlags>().index];
			memory.kind            = ResourceKind::Global;
			memory.component_index = 0;
			memory.component_count = 1;
			const Inserter before {scalar->Parent(), Position(*scalar)};
			scalar->SetArg(3, before(Op::SelectU1, {valid, current, scalar->Arg(3)}));
		}
		const Inserter before {t.record->Parent(), Position(*t.record)};
		t.record->SetArg(4, before(Op::LogicalAnd, {t.record->Arg(4), invalid}));
	}

	// The intersection runs only without the query, whose candidate is a hit at t with u, v.
	const std::array result {part(Op::CompositeExtractU32x4, candidate, 0), Value(One),
	                         part(Op::CompositeExtractU32x4, candidate, 1),
	                         part(Op::CompositeExtractU32x4, candidate, 2)};
	const Inserter   intersect {t.bvh->Parent(), Position(*t.bvh)};
	t.bvh->SetArg(2, intersect(Op::LogicalAnd, {Value(t.exec), invalid}));
	const auto uses = t.bvh->Uses();
	for (const auto& use: uses) {
		auto&          extract = *use.user;
		const Inserter after {extract.Parent(), std::next(Position(extract))};
		const auto     chosen =
		    after(Op::SelectU32, {valid, result[extract.Arg(1).U32()], Value(&extract)});
		Redirect(extract, chosen, *chosen.Instruction());
	}

	// The latch commits a closer hit to the query and takes the next candidate.
	const Inserter vote {t.vote->Parent(), Position(*t.vote)};
	const auto     moved = vote(Op::INotEqual32, {Back(t, *t.extent), Value(t.extent)});
	vote(Op::RayQueryConfirm, {vote(Op::LogicalAnd, {current, moved})});
	const auto alive =
	    vote(Op::SelectU1, {valid, vote(Op::RayQueryProceed, {current}), t.vote->Arg(0)});
	t.vote->SetArg(0, alive);
	for (auto* lane: t.lanes) {
		lane->SetArg(Incoming(*lane, t.latch), alive);
	}
}

// Under the query, the walk skips its body: each iteration takes a lane's next candidate and runs
// copies of the walk's instance and commit tests on it, and a lane leaves at the first it commits.
void RewriteWalk(Program& program, Walk& w) {
	using Op = ValueOpcode;
	const Inserter entry {w.preheader, w.preheader->end()};
	const auto     valid = entry(Op::RayQueryValid, {});
	const auto     start = entry(Op::LogicalAnd, {valid, w.lanes});
	entry(Op::RayQueryInitialize, {w.ray[0], w.ray[1], w.ray[2], w.ray[3], w.ray[4], w.ray[5],
	                               w.ray[6], CullMask(w, Value(w.accepted)), start});
	const Inserter skip {w.skip->Parent(), Position(*w.skip)};
	w.skip->SetArg(0, skip(Op::LogicalAnd, {w.skip->Arg(0), skip(Op::LogicalNot, {valid})}));
	auto& alive = *w.header->PrependNewInst(w.header->begin(), Op::Phi);
	alive.SetFlags(Type::U1);
	alive.AddPhiOperand(w.preheader, start);

	auto*          vote = w.leave->Parent();
	const Inserter at {vote, std::ranges::find_if(*vote, [](const Inst& inst) {
		                   return inst.GetOpcode() != Op::Phi;
	                   })};
	const auto     current   = at(Op::RayQueryProceed, {Value(&alive)});
	const auto     candidate = at(Op::RayQueryCandidate, {current});
	const auto     with      = [&](Op op, Value value, uint32_t immediate) {
		return at(op, {value, Value(immediate)});
	};
	const auto halves = [&](Value address) {
		return std::array {with(Op::CompositeExtractU64, address, 0),
		                   with(Op::CompositeExtractU64, address, 1)};
	};
	// A per-lane load of the guest dword at `offset` from an address.
	const auto load = [&](const std::array<Value, 2>& address, uint32_t offset) {
		const MemoryFlags flags {.index = static_cast<uint32_t>(program.memory_info.size()),
		                         .pc    = w.flags->Flags<MemoryFlags>().pc};
		program.memory_info.push_back(
		    {.kind = ResourceKind::Global, .offset = offset, .address_is_full = true});
		const auto& [low, high] = address;
		return at(Op::LoadAddressU32, {at(Op::GetAddressResource, {low, high}), low, high, current},
		          std::bit_cast<uint64_t>(flags));
	};
	const auto u64 = [&](Value low) { return at(Op::CompositeConstructU64, {low, Value(0u)}); };
	const auto instance = halves(at(Op::RayQueryCandidateInstance, {current}));
	const auto layout   = w.blas.value_or(BlasLayout {});
	const auto blas     = BlasAddress(at, load(instance, layout.offset),
	                                  load(instance, layout.offset + 4u), layout);
	const auto record =
	    halves(at(Op::IAdd64, {blas, with(Op::ShiftLeftLogical64,
		                                  u64(with(Op::CompositeExtractU32x4, candidate, 3)), 3)}));
	// The guest's t numerator and denominator have the sign of its facing, in the node's vertex
	// order, which the mirror reverses for odd remaps of a triangle's barycentric vertices: those
	// of node word 15.
	const auto triangle = load(record, 0);
	const auto vertices =
	    halves(at(Op::IAdd64, {blas, with(Op::ShiftLeftLogical64,
		                                  u64(with(Op::ShiftRightLogical32, triangle, 3)), 6)}));
	const auto shift = with(Op::ShiftLeftLogical32, with(Op::BitwiseAnd32, triangle, 7), 3);
	const auto order =
	    with(Op::BitwiseAnd32, at(Op::ShiftRightLogical32, {load(vertices, 60), shift}), 0xf);
	const auto odd =
	    with(Op::INotEqual32,
		     with(Op::BitwiseAnd32, at(Op::ShiftRightLogical32, {Value(OddRemaps), order}), 1), 0);
	const auto faces = at(Op::LogicalXor, {at(Op::RayQueryCandidateFrontFace, {current}), odd});
	const auto t     = with(Op::CompositeExtractU32x4, candidate, 0);
	const std::array<Value, None> natives {
	    current,
	    at(Op::SelectU32, {faces, t, with(Op::BitwiseOr32, t, 0x80000000u)}),
	    at(Op::SelectU32, {faces, Value(One), Value(0xbf800000u)}),
	    load(instance, InstanceFlags),
	    load(record, 4),
	    instance[0],
	    instance[1]};

	std::unordered_map<const Inst*, Value> copies;
	std::unordered_map<uint32_t, Value>  words;
	const std::function<Value(uint32_t)> word = [&](uint32_t offset) {
		auto& loaded = words[offset];
		if (loaded.IsEmpty()) loaded = load(instance, offset);
		return loaded;
	};
	const auto copy = [&](Value value) { return Copy(w, value, copies, &at, natives, word); };
	const auto accept =
	    at(Op::LogicalAnd,
		   {current, at(Op::LogicalAnd, {copy(Value(w.accepted)), copy(Value(w.commit))})});
	for (const auto& [exit, value]: w.commits) {
		const auto chosen = at(Op::SelectU32, {accept, copy(value), Value(exit)});
		Redirect(*exit, chosen, *chosen.Instruction());
	}
	const auto later = at(Op::LogicalAnd, {current, at(Op::LogicalNot, {accept})});
	alive.AddPhiOperand(w.latch, later);

	// The walk leaves once no lane has a candidate left.
	const Inserter leave {vote, Position(*w.leave)};
	const auto     any =
	    leave(Op::ConditionRef, {later}, static_cast<uint64_t>(CFG::BranchCondition::ExecNonZero));
	w.leave->SetArg(0, leave(Op::SelectU1, {valid, leave(Op::LogicalNot, {any}), w.leave->Arg(0)}));
}

} // namespace

uint32_t TraceTraversalsWithRayQueries(Program& program) {
	if (program.stage != ShaderType::Compute) return 0;
	std::vector<Inst*> intersections;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() == ValueOpcode::BvhIntersect) intersections.push_back(&inst);
		}
	}
	// The blocks of the outermost loops around rewritten traversals, and the intersections left to
	// emulation with the lines their matchers failed on. The other intersections in a rewritten
	// traversal's loops, such as its TLAS walk's and its box tests, are not logged.
	std::vector<std::pair<size_t, size_t>>   rewritten;
	std::vector<std::tuple<Inst*, int, int>> left;
	const auto                               outermost = [&](Block* header) {
		while (auto* up = EnclosingLoop(program, header)) {
			header = up;
		}
		return std::pair {BlockIndex(program, header),
		                  BlockIndex(program, header->terminator.continue_block)};
	};
	uint32_t traced = 0;
	for (auto* bvh: intersections) {
		Traversal   traversal;
		Walk        walk;
		const bool  nested  = bvh->Flags<uint32_t>() == 1u;
		t_fail_line         = 0;
		const bool  matched = nested ? MatchNested(program, *bvh, intersections, traversal)
		                             : Match(program, *bvh, traversal);
		const int   match_fail = t_fail_line;
		t_fail_line         = 0;
		const bool  walked  = !matched && nested && MatchWalk(program, *bvh, intersections, walk);
		const int   walk_fail = t_fail_line;
		const auto& tlas    = walked ? walk.tlas : traversal.tlas;
		if ((matched || walked) && (traced == 0 || tlas == program.info.ray_query_tlas)) {
			rewritten.push_back(outermost(walked ? walk.header : traversal.header));
			walked ? RewriteWalk(program, walk) : Rewrite(program, traversal);
			program.info.ray_query_tlas = tlas;
			++traced;
		} else if (EnclosingLoop(program, bvh->Parent()) != nullptr) {
			left.emplace_back(bvh, match_fail, walk_fail);
		}
	}
	for (const auto& [bvh, match_fail, walk_fail]: left) {
		const auto at = BlockIndex(program, bvh->Parent());
		if (std::ranges::any_of(rewritten, [&](const auto& loops) {
			    return at >= loops.first && at <= loops.second;
		    })) {
			continue;
		}
		Log::WriteToConsoleAndLog(
		    fmt::format("ShaderRecompiler BVH traversal left to emulation: hash=0x{:016x} nested={} "
		                "match failed at line {}, walk at line {}\n",
		                program.shader_hash, bvh->Flags<uint32_t>() == 1u, match_fail, walk_fail));
	}
	return traced;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
