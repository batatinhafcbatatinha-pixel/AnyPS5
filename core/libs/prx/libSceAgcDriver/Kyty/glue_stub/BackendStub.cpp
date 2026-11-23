// The VRAM-resident GPU backend is built on Windows (and on Linux with ANYPS5_KYTY_GPU_LINUX); where it is not, it is never enabled and the
// entry points below are unreachable (callers check Enabled() first).

#include "prx/libSceAgcDriver/Kyty/glue/Backend.hpp"

#include <stdexcept>

namespace AgcDriver::VramBackend {

bool Enabled() {
	return false;
}

void Submit(const Packet*, std::uint32_t) {
	throw std::runtime_error("AGC: the VRAM-resident backend is not available on this host");
}

void SubmitFlip(const FlipInfo&) {
	throw std::runtime_error("AGC: the VRAM-resident backend is not available on this host");
}

std::uint32_t ReserveEopFlip(const FlipInfo&, std::uint32_t) {
	return 0;
}

void RegisterVideoOutput(std::uint32_t, const std::shared_ptr<IVideoOutput>&) {}
void UnregisterVideoOutput(std::uint32_t, const std::shared_ptr<IVideoOutput>&) {}
void AttachWindow(const PresentationWindow&) {}

void Present(const PresentationWindow&, const DisplayBuffer*, const DisplayBuffer*, bool, void (*)(void*), void*) {}

void ReleaseWindow(void*) {}
void RegisterShader(const void*) {}
void SuspendPoint() {}
void WaitIdle() {}
void Shutdown() {}

} // namespace AgcDriver::VramBackend
