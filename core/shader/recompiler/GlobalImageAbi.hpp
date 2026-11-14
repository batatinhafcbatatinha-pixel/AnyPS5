#ifndef CORE_SHADER_RECOMPILER_GLOBALIMAGEABI_HPP
#define CORE_SHADER_RECOMPILER_GLOBALIMAGEABI_HPP

#include <array>
#include <cstdint>

// The global image map of APS5_GPU_SRT (item 3, M3 "bindless images", D:/ps5/perf/REWRITE-STATE.md):
// a GPU-side SRT walk loads an image's T# (8 dwords) or a sampler's S# (4 dwords) at run time and
// finds the bound element for it here instead of in a per-draw descriptor the CPU resolved. The map
// is a storage buffer the driver fills from its texture cache; entries reuse the image-table record
// format (ImageTableAbi.hpp: flags with the runtime mode bits, dword3, element) so the shader samples
// through the existing image-table path. A lookup that finds no entry records the descriptor in the
// miss ring (in the BDA fault buffer, see BdaAbi.hpp) and samples the fallback element.
namespace ShaderRecompiler::GlobalImageAbi {

inline constexpr std::uint32_t Version = 1;

// The device-wide set (Graphics-owned, UPDATE_AFTER_BIND, PARTIALLY_BOUND) of programs with
// RecompileResult::globalImages. Sets 0..3 are taken (RuntimeAbi, WorkgroupMemory, ScratchPool /
// PixelFunctionLds, MeshWindow).
inline constexpr std::uint32_t DescriptorSet = 4;
// Sampled images: one SAMPLED_IMAGE runtime array per view type; an element is written only in the
// array of its view type (the mode the record selects decides which array the shader reads).
// Shaders alias each binding with one variable per sampled type (float / uint / sint / depth).
enum class ViewBinding : std::uint32_t { Image2D = 0, Image2DArray = 1, Image3D = 2, Cube = 3, CubeArray = 4, Image1D = 5, Image1DArray = 6 };
inline constexpr std::uint32_t ViewBindings = 7;
// SAMPLER runtime array; an image-table sampler record's element is the first of
// ImageTableAbi::SamplerEntryElements consecutive variants: +0 filtered, +1 point (integer/converted
// images), +2 compare, +3 compare point.
inline constexpr std::uint32_t SamplerBinding = 7;
// STORAGE_BUFFER map (layout below), read only by shaders.
inline constexpr std::uint32_t MapBinding = 8;

// Map buffer: header, then Capacity entries (open addressing, Probes probes from Hash(key)).
inline constexpr std::uint32_t HeaderVersion = 0;
inline constexpr std::uint32_t HeaderCapacity = 1;  // a power of two
inline constexpr std::uint32_t HeaderFallbackElement = 2;
inline constexpr std::uint32_t HeaderFallbackSampler = 3;
inline constexpr std::uint32_t HeaderWords = 4;

// One entry: the full key (8 words; S# keys use words 0..3 and kind Sampler) so a hash collision can
// never alias, the kind (0: empty), then the record the image-table path consumes.
inline constexpr std::uint32_t EntryKey = 0;
inline constexpr std::uint32_t EntryKeyWords = 8;
inline constexpr std::uint32_t EntryKind = 8;
// Words 9.. hold the image-table record of the kind (ImageTableAbi): images {flags, dword3, element},
// samplers {flags, element}.
inline constexpr std::uint32_t EntryFlags = 9;           // ImageTableAbi record flags
inline constexpr std::uint32_t EntryDword3 = 10;         // images: ImageTableAbi::ImageRecordDword3
inline constexpr std::uint32_t EntryElement = 11;        // images: element in the view-type array
inline constexpr std::uint32_t EntrySamplerElement = 10; // samplers: first of the 4 sampler variants
inline constexpr std::uint32_t EntryWords = 12;

inline constexpr std::uint32_t KindEmpty = 0;
inline constexpr std::uint32_t KindImage = 1;
inline constexpr std::uint32_t KindSampler = 2;

inline constexpr std::uint32_t Probes = 8;

// Image elements (slots) a driver may use; a hit sets the slot's bit in the used-slot bitmap of the
// BDA fault buffer (BdaAbi::GlobalImageUsedWord) so the driver can keep the texture cache's use
// order (eviction) right for textures only shaders read.
inline constexpr std::uint32_t MaxImageSlots = 16384;

// FNV-1a over the key words then the kind, mixed; the shader computes the same in SPIR-V.
constexpr std::uint32_t Hash(const std::array<std::uint32_t, EntryKeyWords>& key, std::uint32_t kind) {
    std::uint32_t hash = 2166136261u;
    for (const auto word : key) {
        hash = (hash ^ word) * 16777619u;
    }
    hash = (hash ^ kind) * 16777619u;
    hash ^= hash >> 15u;
    hash *= 0x2c1b3c6du;
    hash ^= hash >> 12u;
    return hash;
}

constexpr std::uint32_t EntryWord(std::uint32_t entry, std::uint32_t word) {
    return HeaderWords + entry * EntryWords + word;
}

// Key kind word: (signature | 1) where signature = RecompileResult::globalImageClasses[i].signature
// (a hash of the program-side image or sampler and its runtime modes, ClassSignature), so the record
// (mode index into that class's modes) can never be served to a program with another mode list.
// Image keys: the 8 T# words; sampler keys: the 4 S# words, words 4..7 zero.
// Misses go to the BDA fault buffer (BdaAbi.hpp GlobalImageMisses, ForEachGlobalImageMiss), keyed
// by Hash(key, kind).

}

#endif
