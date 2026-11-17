#ifndef APS5_KYTY_GLUE_BACKEND_HPP
#define APS5_KYTY_GLUE_BACKEND_HPP

// The VRAM-resident GPU backend of libSceAgcDriver (KytyPS5's renderer). Default on Windows;
// APS5_GPU_BACKEND=legacy selects the host-import driver instead (GpuBackend::VramResident).

#include "prx/libSceAgcDriver/Execution/include/Presentation.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "SceTypes.hpp"

#include <cstdint>
#include <memory>

namespace AgcDriver::VramBackend {

bool Enabled();
void Submit(const Packet* packet, std::uint32_t queue);
void SubmitFlip(const FlipInfo& info);
// token 0: the next free token.
std::uint32_t ReserveEopFlip(const FlipInfo& info, std::uint32_t token);
void RegisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output);
void UnregisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output);
void AttachWindow(const PresentationWindow& window);
void Present(const PresentationWindow& window, const DisplayBuffer* buffer, const DisplayBuffer* plane,
             bool opaque, void (*gpuReady)(void*), void* context);
void ReleaseWindow(void* window);
void SuspendPoint();
void WaitIdle();
void Shutdown();

} // namespace AgcDriver::VramBackend

#endif
