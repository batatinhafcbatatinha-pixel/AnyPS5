#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIME_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIME_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

// AnyPS5 diagnostics (APS5_KYTY_GPUTIME=1): GPU time per shader program. A bottom-of-pipe timestamp
// follows every draw and dispatch of a main-queue batch (one more opens the batch); the time between
// two timestamps goes to the program of the later one. Every 10 s the GPU thread logs the programs
// that took the most GPU time.
namespace Libs::Graphics {
class RenderContext;
namespace GpuTime {

[[nodiscard]] bool Enabled(RenderContext& context);
// GPU thread, right before recording a draw or dispatch into `command`.
void Before(RenderContext& context, vk::CommandBuffer command);
// GPU thread, right after it: `key` names the program (shader hash), `kind` 'd' draw or 'c' compute,
// upper case when the program walks a BVH, 'q' for a compute program that traces native ray queries.
void After(RenderContext& context, vk::CommandBuffer command, uint64_t key, char kind);
template <typename Program>
[[nodiscard]] char Kind(const Program& program, char kind) {
	if (program.info.ray_query) return 'q';
	return program.info.bvh ? static_cast<char>(kind - 'a' + 'A') : kind;
}

} // namespace GpuTime
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIME_H_
