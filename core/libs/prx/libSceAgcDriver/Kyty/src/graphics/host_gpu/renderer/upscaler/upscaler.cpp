#include "graphics/host_gpu/renderer/upscaler/upscaler.h"

// AnyPS5: the PSSR temporal upscalers (DLSS, FSR) are not part of the backend; PSSR requests take
// RenderExecutor's linear path.

namespace Libs::Graphics::Upscaler {

bool Enabled() {
	return false;
}

std::optional<Dlss::Creation> Choose(const Pssr::Request& /*request*/, const Dlss::Inputs& /*inputs*/) {
	return std::nullopt;
}

std::unique_ptr<Feature> Create(vk::CommandBuffer /*command*/, const Dlss::Creation& /*creation*/) {
	return nullptr;
}

} // namespace Libs::Graphics::Upscaler

#include "graphics/host_gpu/renderer/upscaler/fsr.h"

namespace Libs::Graphics::Dlss {

bool Accepts(const Pssr::Request& /*request*/) {
	return false;
}

bool Takes(const Creation& /*creation*/, const Pssr::Request& /*request*/, const Inputs& /*inputs*/) {
	return false;
}

} // namespace Libs::Graphics::Dlss

namespace Libs::Graphics::Fsr {

bool Enabled() {
	return false;
}

bool Reads(const Pssr::Request& /*request*/, const Dlss::Inputs& /*inputs*/) {
	return false;
}

} // namespace Libs::Graphics::Fsr
