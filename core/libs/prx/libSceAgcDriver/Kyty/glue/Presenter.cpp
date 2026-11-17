// Presentation for the KytyPS5 GPU backend inside AnyPS5.
//
// At a flip the GPU thread copies the display buffer's image (resolved in the texture cache) into
// a frame image, in the flip's own command buffer; libSceVideoOut's presenter thread later blits
// that frame into the swapchain of its window. Frame preparation, the display-buffer resolve and
// the swapchain acquire/submit/present sequence are adapted from KytyPS5
// src/graphics/presentation/window/swapchain.cpp and videoOut.cpp (commit 3a7d8030, GPL-2.0).

#include "glue/CoreInternal.hpp"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <vector>

namespace AgcDriver::VramBackend::Internal {

using namespace Libs::Graphics;

namespace {

struct Frame {
	VulkanImage image;
	bool        blank        = false;
	uint64_t    present_tick = 0;
};

void TransitFrame(vk::CommandBuffer command, VulkanImage& image, vk::ImageLayout layout,
                  vk::AccessFlags2 access, vk::PipelineStageFlags2 stage) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask                    = image.state.pl_stage;
	barrier.srcAccessMask                   = image.state.access_mask;
	barrier.dstStageMask                    = stage;
	barrier.dstAccessMask                   = access;
	barrier.oldLayout                       = image.state.layout;
	barrier.newLayout                       = layout;
	barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                           = image.image;
	barrier.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	barrier.subresourceRange.baseMipLevel   = 0;
	barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
}

void SwapchainBarrier(vk::CommandBuffer command, vk::Image image, vk::ImageLayout from,
                      vk::ImageLayout to, vk::AccessFlags2 src_access, vk::AccessFlags2 dst_access) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask                = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask               = src_access;
	barrier.dstStageMask                = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask               = dst_access;
	barrier.oldLayout                   = from;
	barrier.newLayout                   = to;
	barrier.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                       = image;
	barrier.subresourceRange            = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

vk::Rect2D FitAspect(const vk::Extent3D& source, const vk::Extent2D& target) {
	const uint64_t src_w = std::max(source.width, 1u);
	const uint64_t src_h = std::max(source.height, 1u);
	uint32_t       w     = target.width;
	uint32_t       h     = target.height;
	if (src_w * h > src_h * w) {
		h = std::max<uint32_t>(1, static_cast<uint32_t>(src_h * w / src_w));
	} else {
		w = std::max<uint32_t>(1, static_cast<uint32_t>(src_w * h / src_h));
	}
	return {{static_cast<int32_t>((target.width - w) / 2), static_cast<int32_t>((target.height - h) / 2)},
	        {w, h}};
}

bool DisplayImageInfo(const DisplayDesc& display, ImageInfo& info) {
	const bool compressed  = display.dccAddress != 0;
	const auto compression = ClassifyVideoOutCompression(compressed, display.dccAddress,
	                                                     display.dccControl, display.dccClearColor);
	VideoOutPixelFormatInfo pixel_format {};
	if (display.width == 0 || display.height == 0 || display.address == 0 ||
	    compression == VideoOutCompression::Unsupported ||
	    !DecodeVideoOutPixelFormat(display.pixelFormat, pixel_format)) {
		return false;
	}
	const auto tile_mode =
	    display.tilingMode == 0 ? Prospero::TileMode::kRenderTarget : Prospero::TileMode::kLinear;
	info                 = {};
	info.data            = {display.address, 0};
	info.pixel_format    = pixel_format.format;
	info.guest_format    = pixel_format.guest_format;
	info.type            = Prospero::ImageType::kColor2D;
	info.extent          = {display.width, display.height, 1};
	info.resources       = {1, 1};
	info.bytes_per_block = pixel_format.bytes_per_element;
	info.samples         = 1;
	info.tile_mode       = tile_mode;
	info.bgra16          = pixel_format.bgra16;
	if (display.tilingMode != 0 && display.pitchInPixel != 0) {
		info.pitch = display.pitchInPixel;
	}
	info.UpdateSize();
	if (info.data.size == 0) {
		return false;
	}
	if (compression != VideoOutCompression::Uncompressed) {
		TileSizeAlign dcc_size {};
		if (!TileGetDccSize(display.width, display.height, 1, pixel_format.bytes_per_element, 1,
		                    tile_mode, dcc_size)) {
			return false;
		}
		info.metadata.range       = {display.dccAddress, dcc_size.size};
		info.metadata.kind        = ImageMetadataKind::Dcc;
		info.metadata.compression = compression;
	}
	return IsSupportedVideoOutFormat(info);
}

vk::Format FrameFormat(vk::Format format) {
	switch (format) {
		case vk::Format::eR8G8B8A8Srgb: return vk::Format::eR8G8B8A8Unorm;
		case vk::Format::eB8G8R8A8Srgb: return vk::Format::eB8G8R8A8Unorm;
		default: return format;
	}
}

} // namespace

struct Presenter::Impl {
	explicit Impl(RenderContext& owner)
	    : renderer(owner), graphics(owner.GetGraphics()), present_scheduler(owner, owner.GetGraphics()) {}

	~Impl() {
		std::lock_guard lock(mutex);
		DestroySwapchain();
		for (auto& frame: pool) {
			if (frame->image.image != nullptr) {
				graphics.DeleteImage(frame->image);
			}
		}
		for (auto& [key, frame]: prepared) {
			if (frame->image.image != nullptr) {
				graphics.DeleteImage(frame->image);
			}
		}
		if (surface != nullptr) {
			graphics.instance.destroySurfaceKHR(surface, nullptr);
		}
	}

	std::unique_ptr<Frame> Acquire() {
		std::unique_ptr<Frame> frame;
		{
			std::lock_guard lock(mutex);
			if (!pool.empty()) {
				frame = std::move(pool.back());
				pool.pop_back();
			}
		}
		if (frame == nullptr) {
			return std::make_unique<Frame>();
		}
		present_scheduler.Wait(frame->present_tick);
		return frame;
	}

	void Configure(Frame& frame, vk::Extent2D extent, vk::Format format) {
		auto& image = frame.image;
		if (image.image != nullptr && image.extent.width == extent.width &&
		    image.extent.height == extent.height && image.format == format) {
			return;
		}
		if (image.image != nullptr) {
			graphics.DeleteImage(image);
		}
		vk::ImageCreateInfo create {};
		create.sType         = vk::StructureType::eImageCreateInfo;
		create.imageType     = vk::ImageType::e2D;
		create.extent        = vk::Extent3D {extent.width, extent.height, 1};
		create.mipLevels     = 1;
		create.arrayLayers   = 1;
		create.format        = format;
		create.tiling        = vk::ImageTiling::eOptimal;
		create.initialLayout = vk::ImageLayout::eUndefined;
		create.usage         = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
		create.sharingMode   = vk::SharingMode::eExclusive;
		create.samples       = vk::SampleCountFlagBits::e1;
		if (!graphics.CreateImage(create, image)) {
			EXIT("failed to allocate a presentation frame %ux%u format %d\n", extent.width,
			     extent.height, static_cast<int>(format));
		}
	}

	void DestroySwapchain() {
		if (swapchain == nullptr) {
			return;
		}
		present_scheduler.Finish();
		for (auto semaphore: acquired) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
		for (auto semaphore: rendered) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
		acquired.clear();
		rendered.clear();
		acquired_ticks.clear();
		images.clear();
		graphics.device.destroySwapchainKHR(swapchain, nullptr);
		swapchain = nullptr;
	}

	bool CreateSwapchain(vk::Extent2D drawable) {
		vk::SurfaceCapabilitiesKHR capabilities {};
		if (graphics.physical_device.getSurfaceCapabilitiesKHR(surface, &capabilities) !=
		    vk::Result::eSuccess) {
			return false;
		}
		const auto formats = EnumerateVulkan<vk::SurfaceFormatKHR>(
		    "vkGetPhysicalDeviceSurfaceFormatsKHR", [&](uint32_t* count, vk::SurfaceFormatKHR* values) {
			    return graphics.physical_device.getSurfaceFormatsKHR(surface, count, values);
		    });
		const auto modes = EnumerateVulkan<vk::PresentModeKHR>(
		    "vkGetPhysicalDeviceSurfacePresentModesKHR", [&](uint32_t* count, vk::PresentModeKHR* values) {
			    return graphics.physical_device.getSurfacePresentModesKHR(surface, count, values);
		    });
		if (formats.empty()) {
			return false;
		}
		auto chosen = formats.front();
		for (const auto& candidate: formats) {
			if (candidate.format == vk::Format::eB8G8R8A8Unorm &&
			    candidate.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
				chosen = candidate;
				break;
			}
		}
		// libSceVideoOut paces the flips; the present itself must not wait for a vblank.
		auto mode = vk::PresentModeKHR::eFifo;
		for (const auto preferred: {vk::PresentModeKHR::eImmediate, vk::PresentModeKHR::eMailbox}) {
			if (std::ranges::find(modes, preferred) != modes.end()) {
				mode = preferred;
				break;
			}
		}
		vk::Extent2D extent = capabilities.currentExtent;
		if (extent.width == std::numeric_limits<uint32_t>::max()) {
			extent = drawable;
		}
		extent.width  = std::clamp(extent.width, capabilities.minImageExtent.width,
		                           capabilities.maxImageExtent.width);
		extent.height = std::clamp(extent.height, capabilities.minImageExtent.height,
		                           capabilities.maxImageExtent.height);
		if (extent.width == 0 || extent.height == 0) {
			return false;
		}
		uint32_t count = std::max(capabilities.minImageCount + 1, 3u);
		if (capabilities.maxImageCount != 0) {
			count = std::min(count, capabilities.maxImageCount);
		}
		vk::SwapchainCreateInfoKHR create {};
		create.sType            = vk::StructureType::eSwapchainCreateInfoKHR;
		create.surface          = surface;
		create.minImageCount    = count;
		create.imageFormat      = chosen.format;
		create.imageColorSpace  = chosen.colorSpace;
		create.imageExtent      = extent;
		create.imageArrayLayers = 1;
		create.imageUsage       = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment;
		create.imageSharingMode = vk::SharingMode::eExclusive;
		create.preTransform     = capabilities.currentTransform;
		create.compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque;
		create.presentMode      = mode;
		create.clipped          = VK_TRUE;
		if (graphics.device.createSwapchainKHR(&create, nullptr, &swapchain) != vk::Result::eSuccess) {
			swapchain = nullptr;
			return false;
		}
		images = EnumerateVulkan<vk::Image>("vkGetSwapchainImagesKHR", [&](uint32_t* number, vk::Image* values) {
			return graphics.device.getSwapchainImagesKHR(swapchain, number, values);
		});
		swap_format = chosen.format;
		swap_extent = extent;
		window_extent = drawable;
		for (size_t i = 0; i < images.size(); i++) {
			vk::SemaphoreCreateInfo info {};
			info.sType = vk::StructureType::eSemaphoreCreateInfo;
			vk::Semaphore first  = nullptr;
			vk::Semaphore second = nullptr;
			RequireVulkanSuccess(graphics.device.createSemaphore(&info, nullptr, &first), "vkCreateSemaphore");
			RequireVulkanSuccess(graphics.device.createSemaphore(&info, nullptr, &second), "vkCreateSemaphore");
			acquired.push_back(first);
			rendered.push_back(second);
			acquired_ticks.push_back(0);
		}
		frame_index = 0;
		return true;
	}

	RenderContext&                         renderer;
	GraphicContext&                        graphics;
	CommandScheduler                       present_scheduler;
	std::mutex                             mutex;
	std::vector<std::unique_ptr<Frame>>    pool;
	std::map<void*, std::unique_ptr<Frame>> prepared;
	vk::SurfaceKHR                         surface = nullptr;
	void*                                  window  = nullptr;
	vk::SwapchainKHR                       swapchain = nullptr;
	vk::Format                             swap_format = vk::Format::eUndefined;
	vk::Extent2D                           swap_extent {};
	vk::Extent2D                           window_extent {};
	std::vector<vk::Image>                 images;
	std::vector<vk::Semaphore>             acquired;
	std::vector<vk::Semaphore>             rendered;
	std::vector<uint64_t>                  acquired_ticks;
	uint32_t                               frame_index = 0;
};

Presenter::Presenter(RenderContext& renderer): m_impl(std::make_unique<Impl>(renderer)) {}

Presenter::~Presenter() = default;

void Presenter::PrepareFrame(CommandBuffer& command_buffer, void* key, const DisplayDesc& display) {
	ImageInfo info {};
	if (!DisplayImageInfo(display, info)) {
		static std::once_flag reported;
		std::call_once(reported, [&] {
			Log::WriteToConsoleAndLog(fmt::format(
			    "unsupported display buffer (format 0x{:x}, tiling {}, DCC 0x{:x}); its flips show black\n",
			    display.pixelFormat, display.tilingMode, display.dccControl));
		});
		PrepareBlank(key);
		return;
	}
	auto                     frame = m_impl->Acquire();
	TextureCache::ImageDesc desc {};
	desc.info                  = info;
	desc.view_info.format      = info.pixel_format;
	desc.view_info.type        = vk::ImageViewType::e2D;
	desc.view_info.aspect      = vk::ImageAspectFlagBits::eColor;
	desc.view_info.base_level  = 0;
	desc.view_info.level_count = 1;
	desc.view_info.base_layer  = 0;
	desc.view_info.layer_count = 1;
	desc.view_info.usage       = vk::ImageUsageFlagBits::eTransferSrc;
	desc.type                  = TextureCache::BindingType::VideoOut;
	{
		Common::LockGuard render_lock(m_impl->renderer.GetMutex());
		auto&             cache    = m_impl->renderer.GetTextureCache();
		const auto        image_id = cache.FindImage(desc);
		auto&             image    = cache.GetImage(image_id);
		image.usage.video_out      = true;
		cache.UpdateImage(image_id);
		if (image.backing.format == vk::Format::eUndefined) {
			EXIT("unsupported presentation source image\n");
		}
		m_impl->Configure(*frame, {image.backing.extent.width, image.backing.extent.height},
		                  FrameFormat(info.pixel_format));
		command_buffer.EndRendering();
		auto command = command_buffer.Handle();
		image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
		              command);
		TransitFrame(command, frame->image, vk::ImageLayout::eTransferDstOptimal,
		             vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
		vk::ImageCopy copy {};
		copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copy.extent         = vk::Extent3D {std::min(image.backing.extent.width, frame->image.extent.width),
		                                    std::min(image.backing.extent.height, frame->image.extent.height), 1};
		command.copyImage(image.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  frame->image.image, vk::ImageLayout::eTransferDstOptimal, copy);
		TransitFrame(command, frame->image, vk::ImageLayout::eTransferSrcOptimal,
		             vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer);
	}
	frame->blank = false;
	std::lock_guard lock(m_impl->mutex);
	if (auto previous = m_impl->prepared.find(key); previous != m_impl->prepared.end()) {
		m_impl->pool.push_back(std::move(previous->second));
		m_impl->prepared.erase(previous);
	}
	m_impl->prepared.emplace(key, std::move(frame));
}

void Presenter::PrepareBlank(void* key) {
	auto frame   = std::make_unique<Frame>();
	frame->blank = true;
	std::lock_guard lock(m_impl->mutex);
	m_impl->prepared[key] = std::move(frame);
}

void Presenter::AttachSurface(vk::SurfaceKHR surface, void* window) {
	std::lock_guard lock(m_impl->mutex);
	m_impl->DestroySwapchain();
	if (m_impl->surface != nullptr) {
		m_impl->graphics.instance.destroySurfaceKHR(m_impl->surface, nullptr);
	}
	m_impl->surface = surface;
	m_impl->window  = window;
	if (surface != nullptr && !SurfaceSupportsPresent(m_impl->graphics, surface)) {
		Log::WriteToConsoleAndLog("the window's surface cannot present from the GPU queue family; nothing is shown\n");
		m_impl->graphics.instance.destroySurfaceKHR(surface, nullptr);
		m_impl->surface = nullptr;
	}
}

void Presenter::ReleaseWindow(void* window) {
	std::lock_guard lock(m_impl->mutex);
	if (m_impl->window != window) {
		return;
	}
	m_impl->DestroySwapchain();
	if (m_impl->surface != nullptr) {
		m_impl->graphics.instance.destroySurfaceKHR(m_impl->surface, nullptr);
		m_impl->surface = nullptr;
	}
	m_impl->window = nullptr;
}

void Presenter::Present(const WindowDesc& window, void* key, bool opaque) {
	auto&                  impl = *m_impl;
	std::unique_ptr<Frame> frame;
	std::lock_guard        lock(impl.mutex);
	if (key != nullptr) {
		if (const auto found = impl.prepared.find(key); found != impl.prepared.end()) {
			frame = std::move(found->second);
			impl.prepared.erase(found);
		}
	}
	const auto give_back = [&] {
		if (frame != nullptr && frame->image.image != nullptr) {
			impl.pool.push_back(std::move(frame));
		}
	};
	if (impl.surface == nullptr || window.context != impl.window) {
		give_back();
		return;
	}
	vk::Extent2D drawable {};
	if (window.getDrawableSize != nullptr) {
		window.getDrawableSize(window.context, &drawable.width, &drawable.height);
	}
	if (impl.swapchain != nullptr &&
	    (drawable.width != impl.window_extent.width || drawable.height != impl.window_extent.height)) {
		impl.DestroySwapchain();
	}
	if (impl.swapchain == nullptr && !impl.CreateSwapchain(drawable)) {
		give_back();
		return;
	}
	const auto slot = impl.frame_index;
	impl.present_scheduler.Wait(impl.acquired_ticks[slot]);
	uint32_t   image_index = 0;
	const auto acquired    = impl.graphics.device.acquireNextImageKHR(
        impl.swapchain, std::numeric_limits<uint64_t>::max(), impl.acquired[slot], nullptr, &image_index);
	if (acquired != vk::Result::eSuccess && acquired != vk::Result::eSuboptimalKHR) {
		impl.DestroySwapchain();
		give_back();
		return;
	}
	auto& command_buffer = impl.present_scheduler.BeginCommand();
	auto  command        = command_buffer.Handle();
	auto  target         = impl.images[image_index];
	SwapchainBarrier(command, target, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
	                 vk::AccessFlagBits2::eNone, vk::AccessFlagBits2::eTransferWrite);
	vk::ClearColorValue clear {};
	clear.float32[3] = opaque ? 1.0f : 0.0f;
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	command.clearColorImage(target, vk::ImageLayout::eTransferDstOptimal, &clear, 1, &range);
	if (frame != nullptr && !frame->blank && frame->image.image != nullptr) {
		const auto rect = FitAspect(frame->image.extent, impl.swap_extent);
		vk::ImageBlit blit {};
		blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(frame->image.extent.width),
		                                    static_cast<int32_t>(frame->image.extent.height), 1};
		blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		blit.dstOffsets[0]  = vk::Offset3D {rect.offset.x, rect.offset.y, 0};
		blit.dstOffsets[1]  = vk::Offset3D {rect.offset.x + static_cast<int32_t>(rect.extent.width),
		                                    rect.offset.y + static_cast<int32_t>(rect.extent.height), 1};
		SwapchainBarrier(command, target, vk::ImageLayout::eTransferDstOptimal,
		                 vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
		                 vk::AccessFlagBits2::eTransferWrite);
		command.blitImage(frame->image.image, vk::ImageLayout::eTransferSrcOptimal, target,
		                  vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);
	}
	SwapchainBarrier(command, target, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
	                 vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eNone);
	SubmitInfo submit;
	submit.AddWait(impl.acquired[slot], 1, vk::PipelineStageFlagBits::eTransfer);
	submit.AddSignal(impl.rendered[image_index]);
	const auto tick           = impl.present_scheduler.Submit(submit);
	impl.acquired_ticks[slot] = tick;
	if (frame != nullptr) {
		frame->present_tick = tick;
	}
	vk::PresentInfoKHR present {};
	present.sType              = vk::StructureType::ePresentInfoKHR;
	present.swapchainCount     = 1;
	present.pSwapchains        = &impl.swapchain;
	present.pImageIndices      = &image_index;
	present.pWaitSemaphores    = &impl.rendered[image_index];
	present.waitSemaphoreCount = 1;
	vk::Result presented;
	{
		Common::LockGuard queue_lock(impl.graphics.queue_mutex);
		presented = impl.graphics.queue.presentKHR(&present);
	}
	impl.frame_index = (impl.frame_index + 1u) % static_cast<uint32_t>(impl.images.size());
	give_back();
	if (presented != vk::Result::eSuccess) {
		impl.DestroySwapchain();
	}
}

} // namespace AgcDriver::VramBackend::Internal
