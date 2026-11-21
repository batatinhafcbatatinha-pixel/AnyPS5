#ifndef CORE_SHADER_RECOMPILER_BDAABI_HPP
#define CORE_SHADER_RECOMPILER_BDAABI_HPP

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace ShaderRecompiler::BdaAbi {

inline constexpr std::uint32_t Version = 1;
inline constexpr std::uint32_t Read = 1;
inline constexpr std::uint32_t Write = 2;
// Header::reserved bit 0: the buffer cache's two-level page table root (a u64 device address) follows
// the last Range. Read-only lookups then try root[A >> CacheLeafShift] -> leaf,
// leaf[(A & ((1 << CacheLeafShift) - 1)) >> CachePageShift] -> page device address (0: not resident)
// before the range search. Must match Graphics/include/BufferCache.hpp.
inline constexpr std::uint32_t CacheRootFlag = 1;
inline constexpr std::uint32_t CachePageShift = 14;
inline constexpr std::uint32_t CacheLeafShift = 30;
inline constexpr std::uint64_t CacheRoots = 1024;

struct Header {
    std::uint32_t version;
    std::uint32_t count;
    std::uint32_t entryBytes;
    std::uint32_t reserved;
};

struct Range {
    std::uint64_t begin;
    std::uint64_t end;
    std::uint64_t deviceAddress;
    std::uint32_t permissions;
    std::uint32_t reserved;
};

inline bool RangesOrdered(const Range* ranges, std::size_t count) {
    for (std::size_t index = 1; index < count; ++index) {
        if (ranges[index].begin < ranges[index - 1].begin) return false;
    }
    return true;
}

enum class FaultState : std::uint32_t { Empty, Writing, Ready };
enum class FaultReason : std::uint32_t { Unmapped = 1, Permission, Overflow, InvalidTable, InvalidRectangle, LoopLimit, Unaligned, InvalidDescriptor, ImageTable, UnlinkedCall, WindowLaneSpill, SplitState, StaleCache };

struct Fault {
    FaultState state;
    FaultReason reason;
    std::uint64_t address;
    std::uint32_t bytes;
    std::uint32_t stage;
    std::uint32_t instruction;
    std::uint32_t claim;
    std::uint32_t discard;
    std::uint32_t reserved;
};

inline constexpr std::uint32_t FaultClaimed = 1u;
inline constexpr std::uint32_t FaultClaimOtherFaults = 2u;

inline constexpr std::uint32_t OtherFaults(const Fault& fault) {
    return (fault.claim & FaultClaimOtherFaults) != 0u ? 1u : 0u;
}

static_assert(std::is_standard_layout_v<Header> && std::is_trivially_copyable_v<Header> && sizeof(Header) == 16);
static_assert(std::is_standard_layout_v<Range> && std::is_trivially_copyable_v<Range> && sizeof(Range) == 32);
static_assert(offsetof(Range, deviceAddress) == 16 && offsetof(Range, permissions) == 24);
static_assert(std::is_standard_layout_v<Fault> && std::is_trivially_copyable_v<Fault> && sizeof(Fault) == 40);
static_assert(offsetof(Fault, address) == 8 && offsetof(Fault, instruction) == 24 && offsetof(Fault, claim) == 28 && offsetof(Fault, discard) == 32);

inline constexpr std::uint32_t WrittenPageShift = 12;
inline constexpr std::uint32_t WrittenChunkShift = 17;
inline constexpr std::uint32_t WrittenChunkPages = 1u << (WrittenChunkShift - WrittenPageShift);
inline constexpr std::uint32_t WrittenPageSlots = 2048;
inline constexpr std::uint32_t WrittenPageProbes = 8;
inline constexpr std::uint64_t WrittenAddressLimit = std::uint64_t{0xffffffffu} << WrittenChunkShift;
inline constexpr std::uint32_t WrittenOverflowWord = sizeof(Fault) / sizeof(std::uint32_t);
inline constexpr std::uint32_t WrittenCountWord = WrittenOverflowWord + 1;
inline constexpr std::uint32_t WrittenSlotsWord = WrittenCountWord + 1;
inline constexpr std::uint32_t WrittenListWord = WrittenSlotsWord + 2 * WrittenPageSlots;
inline constexpr std::uint32_t SrtGuardWord = WrittenListWord + WrittenPageSlots;
inline constexpr std::uint32_t SrtGuardStageShift = 24;
// Accessed pages (APS5_GPU_SRT with the buffer cache root present): a read lookup that missed the
// cache page table but resolved through a range notes its 2^CachePageShift-byte page here: key
// (A >> CachePageShift) + 1 claimed in AccessedSlotsWord (open addressing, AccessedPageProbes probes),
// the claimed slot index appended at AccessedListWord, AccessedCountWord counting them. Graphics makes
// the pages resident after the batch and clears the region (ForEachAccessedPage).
inline constexpr std::uint32_t AccessedPageSlots = 512;
inline constexpr std::uint32_t AccessedPageProbes = 8;
inline constexpr std::uint32_t AccessedCountWord = SrtGuardWord + 1;
inline constexpr std::uint32_t AccessedSlotsWord = AccessedCountWord + 1;
inline constexpr std::uint32_t AccessedListWord = AccessedSlotsWord + AccessedPageSlots;
// Global image misses (APS5_GPU_SRT=4, GlobalImageAbi.hpp): a shader whose T#/S# key is not in the
// global image map claims a slot keyed by the key's hash: MissClaimsWord + (hash + probe) &
// (GlobalImageMisses - 1) for GlobalImageMissProbes probes, CAS 0 -> (hash | 1); the lane that
// claimed it stores kind + 8 key words at MissRecordsWord + slot * GlobalImageMissRecordWords and
// counts MissCountWord. Lanes finding their own hash in a slot stop (the same key, or a colliding one
// noted again in a later batch). Graphics reads it after the batch (ForEachGlobalImageMiss).
inline constexpr std::uint32_t GlobalImageMisses = 256;
inline constexpr std::uint32_t GlobalImageMissProbes = 8;
inline constexpr std::uint32_t GlobalImageMissRecordWords = 9;
inline constexpr std::uint32_t MissCountWord = AccessedListWord + AccessedPageSlots;
inline constexpr std::uint32_t MissClaimsWord = MissCountWord + 1;
inline constexpr std::uint32_t MissRecordsWord = MissClaimsWord + GlobalImageMisses;
// Global image slots sampled by the batch (GlobalImageAbi::MaxImageSlots bits): a hit sets its slot's
// bit (a plain load first, an atomic OR only when clear); Graphics reads and clears it after the batch
// (ForEachUsedGlobalImageSlot).
inline constexpr std::uint32_t GlobalImageUsedWords = 16384u / 32u;
inline constexpr std::uint32_t GlobalImageUsedWord = MissRecordsWord + GlobalImageMisses * GlobalImageMissRecordWords;
// APS5_GPU_SRT_GLOBAL_COUNT (diagnostic): subgroups whose first lane's global image lookup hit /
// missed, then the same for samplers (4 words), read and cleared by Graphics after the batch.
inline constexpr std::uint32_t GlobalImageCountWord = GlobalImageUsedWord + GlobalImageUsedWords;
inline constexpr std::uint32_t GlobalImageCountWords = 4;
inline constexpr std::size_t FaultBufferBytes = (GlobalImageCountWord + GlobalImageCountWords) * sizeof(std::uint32_t);
static_assert((AccessedPageSlots & (AccessedPageSlots - 1)) == 0);
static_assert((GlobalImageMisses & (GlobalImageMisses - 1)) == 0);
static_assert((WrittenPageSlots & (WrittenPageSlots - 1)) == 0);
static_assert(WrittenChunkPages == 32);

inline constexpr std::uint32_t WrittenChunkKey(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> WrittenChunkShift) + 1u;
}

// Calls visit(page address) for each noted accessed page and clears the region. False when the count
// was out of range (the region is still visited and cleared).
template<typename TVisit>
bool ForEachAccessedPage(std::uint32_t* words, std::uint32_t pageShift, TVisit&& visit) {
    // The count is bounded by the slots (a claim is a successful CAS on an empty slot), but a stale or
    // corrupted region must not stick: every slot is visited and cleared whatever the count says.
    const bool wellFormed = words[AccessedCountWord] <= AccessedPageSlots;
    for (std::uint32_t slot = 0; slot < AccessedPageSlots; ++slot) {
        const auto key = words[AccessedSlotsWord + slot];
        if (key == 0) continue;
        visit(static_cast<std::uint64_t>(key - 1u) << pageShift);
        words[AccessedSlotsWord + slot] = 0;
    }
    for (std::uint32_t claim = 0; claim < AccessedPageSlots; ++claim) words[AccessedListWord + claim] = 0;
    words[AccessedCountWord] = 0;
    return wellFormed;
}

// Calls visit(kind, key) (key: 8 words) for each noted global image miss and clears the region.
template<typename TVisit>
void ForEachGlobalImageMiss(std::uint32_t* words, TVisit&& visit) {
    if (words[MissCountWord] == 0) return;
    for (std::uint32_t slot = 0; slot < GlobalImageMisses; ++slot) {
        if (words[MissClaimsWord + slot] == 0) continue;
        const auto* record = words + MissRecordsWord + slot * GlobalImageMissRecordWords;
        if (record[0] != 0) visit(record[0], std::span<const std::uint32_t, 8>(record + 1, 8));
        words[MissClaimsWord + slot] = 0;
        for (std::uint32_t word = 0; word < GlobalImageMissRecordWords; ++word) words[MissRecordsWord + slot * GlobalImageMissRecordWords + word] = 0;
    }
    words[MissCountWord] = 0;
}

// Calls visit(slot) for each global image slot marked used and clears the bitmap.
template<typename TVisit>
void ForEachUsedGlobalImageSlot(std::uint32_t* words, TVisit&& visit) {
    for (std::uint32_t word = 0; word < GlobalImageUsedWords; ++word) {
        auto bits = words[GlobalImageUsedWord + word];
        if (bits == 0) continue;
        words[GlobalImageUsedWord + word] = 0;
        while (bits != 0) {
            const auto bit = static_cast<std::uint32_t>(std::countr_zero(bits));
            visit(word * 32u + bit);
            bits &= bits - 1u;
        }
    }
}

template<typename TMark>
bool ForEachWrittenRange(const std::uint32_t* words, TMark&& mark) {
    const auto count = words[WrittenCountWord];
    if (count > WrittenPageSlots) return false;
    for (std::uint32_t claim = 0; claim < count; ++claim) {
        const auto slot = words[WrittenListWord + claim];
        if (slot >= WrittenPageSlots || words[WrittenSlotsWord + 2 * slot] == 0) return false;
    }
    for (std::uint32_t claim = 0; claim < count; ++claim) {
        const auto slot = words[WrittenListWord + claim];
        const auto key = words[WrittenSlotsWord + 2 * slot];
        auto mask = words[WrittenSlotsWord + 2 * slot + 1];
        const auto base = static_cast<std::uint64_t>(key - 1u) << WrittenChunkShift;
        while (mask != 0) {
            const auto first = static_cast<std::uint32_t>(std::countr_zero(mask));
            const auto pages = static_cast<std::uint32_t>(std::countr_one(mask >> first));
            mark(base + (static_cast<std::uint64_t>(first) << WrittenPageShift), static_cast<std::size_t>(pages) << WrittenPageShift);
            mask = pages + first >= 32u ? 0u : mask & ~((1u << (first + pages)) - 1u);
        }
    }
    return true;
}

}

#endif
