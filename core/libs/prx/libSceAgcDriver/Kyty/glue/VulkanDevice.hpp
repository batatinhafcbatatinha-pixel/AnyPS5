#ifndef APS5_KYTY_GLUE_VULKANDEVICE_HPP
#define APS5_KYTY_GLUE_VULKANDEVICE_HPP

#include "graphics/host_gpu/vulkanCommon.h"

#include <span>

namespace Libs::Graphics {

struct GraphicContext;

// Instance with the surface extensions (and the window's), Vulkan 1.3, Vulkan-Hpp dispatcher.
void CreateBackendInstance(GraphicContext& graphics, PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                           std::span<const char* const> window_extensions);
// Physical device, device, queues and VMA allocator; `surface` may be null (no window yet).
void CreateBackendDevice(GraphicContext& graphics, vk::SurfaceKHR surface);
[[nodiscard]] bool SurfaceSupportsPresent(const GraphicContext& graphics, vk::SurfaceKHR surface);
void DestroyBackendDevice(GraphicContext& graphics);

} // namespace Libs::Graphics

#endif
