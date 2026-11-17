#ifndef APS5_KYTY_GLUE_COREINTERNAL_HPP
#define APS5_KYTY_GLUE_COREINTERNAL_HPP

#include "glue/Core.hpp"

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <memory>

namespace Libs::Graphics {
class CommandBuffer;
class RenderContext;
struct GraphicContext;
} // namespace Libs::Graphics

namespace AgcDriver::VramBackend::Internal {

// Null until Core::Start created the device.
Libs::Graphics::RenderContext* Renderer();
Libs::Graphics::GraphicContext& Graphics();

// Frames copied from display buffers on the GPU thread and shown by the presenter thread.
class Presenter {
public:
	explicit Presenter(Libs::Graphics::RenderContext& renderer);
	~Presenter();
	Presenter(const Presenter&)            = delete;
	Presenter& operator=(const Presenter&) = delete;

	// GPU thread, inside the flip's command buffer.
	void PrepareFrame(Libs::Graphics::CommandBuffer& command, void* key, const DisplayDesc& display);
	void PrepareBlank(void* key);
	// Presenter thread.
	void Present(const WindowDesc& window, void* key, bool opaque);
	void AttachSurface(vk::SurfaceKHR surface, void* window);
	void ReleaseWindow(void* window);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

Presenter* CurrentPresenter();

} // namespace AgcDriver::VramBackend::Internal

#endif
