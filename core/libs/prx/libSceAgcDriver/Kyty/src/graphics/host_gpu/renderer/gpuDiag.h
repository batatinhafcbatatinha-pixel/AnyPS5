#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUDIAG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUDIAG_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

// AnyPS5 diagnostics (APS5_KYTY_GPU_DIAG=1) for device losses and GPU hangs: the device is created
// with VK_NV_device_diagnostic_checkpoints, VK_NV_device_diagnostics_config and VK_EXT_device_fault
// when it has them, every draw and dispatch of the GPU thread records a checkpoint naming its
// program, and a loss (or a GPU wait that does not end) logs the last checkpoints each queue reached
// and the device's fault report (faulting addresses and vendor data).
namespace Libs::Graphics {
struct GraphicContext;
namespace GpuDiag {

[[nodiscard]] bool Wanted();
// GPU thread, before recording a draw or dispatch: `key` names the program, `kind` 'd' or 'c'.
void Checkpoint(const GraphicContext& graphics, vk::CommandBuffer command, uint64_t key, char kind,
                uint64_t tick);
// Any thread, after a device loss or a stuck GPU wait; logs once per call site kind.
void Report(const GraphicContext& graphics, const char* where);

} // namespace GpuDiag
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUDIAG_H_
