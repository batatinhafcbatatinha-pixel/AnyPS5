#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICELOSSREPORT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICELOSSREPORT_H_

namespace Libs::Graphics {

struct GraphicContext;

// One log line when a Vulkan call fails (device loss): host RAM, commit charge and VRAM
// budget/usage, so a loss caused by memory pressure can be told from a GPU fault.
void ReportDeviceLoss(const GraphicContext& graphics, const char* where, int result);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICELOSSREPORT_H_
