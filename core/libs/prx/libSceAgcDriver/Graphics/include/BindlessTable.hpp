#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BINDLESSTABLE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BINDLESSTABLE_HPP


#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

class Recorder;

constexpr std::uint32_t BindlessImageSlots = 16384;
constexpr std::uint32_t BindlessSamplerSlots = 2048;
constexpr std::uint32_t BindlessMapEntries = 32768;

bool BindlessEnabled(const Context& context);
VkDescriptorSetLayout BindlessSetLayout(const Context& context);
VkDescriptorSet BindlessSet(const Context& context);
void QueueBindlessMisses(VkDevice device, std::uint32_t* faultWords);
void FlushBindless(const Context& context, Recorder& recorder);
// The GPU-backed textures global-image shaders sampled recently, as the reads of the next draw: images
// (storage images they view; images listed as written, the draw's attachments, are skipped) and guest
// ranges (other GPU-backed views, unless they overlap an excluded range: the draw's own depth).
void BindlessDrawReads(const Context& context, std::vector<std::pair<VkImage, bool>>& images, std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges, std::span<const std::pair<std::uint64_t, std::uint64_t>> excluded);

}

#endif
