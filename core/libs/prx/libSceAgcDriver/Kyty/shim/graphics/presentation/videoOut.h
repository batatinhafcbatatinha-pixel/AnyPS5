#ifndef APS5_KYTY_SHIM_GRAPHICS_PRESENTATION_VIDEOOUT_H
#define APS5_KYTY_SHIM_GRAPHICS_PRESENTATION_VIDEOOUT_H

// The video-out side the GPU stack calls into. In AnyPS5 it bridges to libSceVideoOut's
// IVideoOutput / IFlipRequest (glue/VideoOutBridge.cpp) instead of owning the flip queue.

#include "common/common.h"
#include "libs/errno.h"

#include <cstdint>
#include <memory>

namespace Libs::Graphics {
class CommandBuffer;
} // namespace Libs::Graphics

namespace Libs::VideoOut {

class VideoOutDriver final {
public:
	struct Impl;

	VideoOutDriver();
	~VideoOutDriver();
	KYTY_CLASS_NO_COPY(VideoOutDriver);

	// GPU thread, at a flip in the command stream: reserves the flip and records the copy of its
	// display buffer into the current command buffer.
	int  SubmitFlipFromGpu(Graphics::CommandBuffer& buffer, int handle, int index, int flip_mode,
	                       int64_t flip_arg, uint64_t& request_id);
	void PrepareFlip(uint64_t request_id, Graphics::CommandBuffer& buffer);
	// After the batch holding the copy completed: hands the frame to the presenter.
	void CompleteFlip(uint64_t request_id);
	void WaitForSubmitSlot(int handle);
	void WaitFlipDone(const uint64_t* label);

	// AnyPS5 extensions: an end-of-pipe release whose interrupt context is a reserved flip token
	// flips (true, request reserved); the driver's rendering wait packet.
	bool TakeEopFlip(uint32_t token, Graphics::CommandBuffer& buffer, uint64_t& request_id);
	void RenderingWait(uint32_t handle, uint32_t index);

	[[nodiscard]] Impl& State() noexcept { return *m_impl; }

private:
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::VideoOut

#endif
