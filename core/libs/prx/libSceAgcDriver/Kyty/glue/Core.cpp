// Lifecycle of the KytyPS5 GPU stack inside AnyPS5's libSceAgcDriver: device creation, the guest
// GPU thread, command submission and the video-out bridge the stack calls at flips.

#include "common/frameClock.h"
#include "glue/Core.hpp"
#include "glue/CoreInternal.hpp"
#include "glue/VulkanDevice.hpp"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "libs/errno.h"

#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/HostFault.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <link.h>
#endif

#include <algorithm>
#include <atomic>
#include <mutex>
#include <span>

namespace AgcDriver::VramBackend {

namespace {

std::recursive_mutex                          g_mutex;
Libs::Graphics::GraphicContext                g_graphics;
std::unique_ptr<Libs::Graphics::RenderContext> g_renderer;
std::unique_ptr<Libs::VideoOut::VideoOutDriver> g_video_out;
std::unique_ptr<Internal::Presenter>          g_presenter;
std::atomic<Libs::Graphics::RenderContext*>    g_published {nullptr};
#ifdef _WIN32
HMODULE                                       g_vulkan = nullptr;
#else
void*                                         g_vulkan = nullptr;
#endif

#ifdef _WIN32
bool FaultThunk(void* exception_pointers) noexcept {
	return Core::HandleFault(exception_pointers);
}
#else
bool FaultThunk(void* signal_info, void* signal_context) noexcept {
	return Core::HandleFault(signal_info, signal_context);
}
#endif

void MappedThunk(std::uintptr_t address, std::size_t bytes) {
	Core::NoteMapped(address, bytes);
}

void ReleasedThunk(std::uintptr_t address, std::size_t bytes) {
	Core::NoteReleased(address, bytes);
}

void HostWriteThunk(std::uintptr_t address, std::size_t bytes) {
	Core::NoteHostWrite(address, bytes);
}

void HostReadThunk(std::uintptr_t address, std::size_t bytes) {
	Core::NoteHostRead(address, bytes);
}

// SRT-walker loads of the current guest word without touching a protected page (KytyPS5
// src/kernel/memory.cpp WalkerFastLoad, commit 3a7d8030).
uint64_t WalkerFastLoad(uint64_t vaddr) {
	auto* renderer = g_published.load(std::memory_order_acquire);
	uint32_t value = 0;
	if ((vaddr & 3u) != 0 || renderer == nullptr || !Libs::Graphics::GuestGpu::IsGpuThread() ||
	    !renderer->GetBufferCache().TryCurrentWalkerLoad(vaddr, value)) {
		return 0;
	}
	return (uint64_t {1} << 32u) | value;
}

vk::SurfaceKHR CreateSurface(const WindowDesc& window) {
	if (window.createSurface == nullptr || window.context == nullptr) {
		return nullptr;
	}
	const auto handle = window.createSurface(
	    window.context, static_cast<void*>(static_cast<VkInstance>(g_graphics.instance)));
	return vk::SurfaceKHR(reinterpret_cast<VkSurfaceKHR>(handle));
}

// The guest arena and the relinked executable: every range the GPU may be handed.
void MapGuestRanges(Libs::Graphics::RenderContext& renderer) {
	std::uintptr_t base  = 0;
	std::size_t    bytes = 0;
	GuestArena::GuestArenaRange_nid_postfix(&base, &bytes);
	const auto lower = std::min<uint64_t>(base + bytes, Libs::Graphics::LOWER_ADDRESS_SIZE);
	if (base != 0 && lower > base) {
		renderer.MapMemory(base, lower - base);
	}
#ifdef _WIN32
	const auto* image = reinterpret_cast<const std::byte*>(GetModuleHandleW(nullptr));
	const auto* dos   = reinterpret_cast<const IMAGE_DOS_HEADER*>(image);
	const auto* nt    = reinterpret_cast<const IMAGE_NT_HEADERS64*>(image + dos->e_lfanew);
	const auto  start = reinterpret_cast<uint64_t>(image);
	const auto  size  = static_cast<uint64_t>(nt->OptionalHeader.SizeOfImage);
#else
	// The relinked executable is the first object dl_iterate_phdr reports: its loadable segments
	// from the lowest to the highest address.
	struct Span {
		uint64_t low  = UINT64_MAX;
		uint64_t high = 0;
	} span;
	dl_iterate_phdr(
	    [](dl_phdr_info* info, size_t, void* data) -> int {
		    auto* out = static_cast<Span*>(data);
		    for (int index = 0; index < info->dlpi_phnum; ++index) {
			    const auto& header = info->dlpi_phdr[index];
			    if (header.p_type != PT_LOAD) {
				    continue;
			    }
			    const uint64_t low  = (info->dlpi_addr + header.p_vaddr) & ~uint64_t {0xfff};
			    const uint64_t high = (info->dlpi_addr + header.p_vaddr + header.p_memsz + 0xfff) & ~uint64_t {0xfff};
			    out->low            = std::min(out->low, low);
			    out->high           = std::max(out->high, high);
		    }
		    return 1; // the main program only
	    },
	    &span);
	const uint64_t start = span.low;
	const uint64_t size  = span.high > span.low ? span.high - span.low : 0;
#endif
	if (size != 0 && Libs::Graphics::GuestRange {start, size}.Valid()) {
		renderer.MapMemory(start, size);
	}
}

} // namespace

namespace Internal {

Libs::Graphics::RenderContext* Renderer() {
	return g_published.load(std::memory_order_acquire);
}

Libs::Graphics::GraphicContext& Graphics() {
	return g_graphics;
}

Presenter* CurrentPresenter() {
	return g_presenter.get();
}

} // namespace Internal

bool Core::Started() {
	return g_published.load(std::memory_order_acquire) != nullptr;
}

void Core::Start(const WindowDesc* window) {
	std::lock_guard lock(g_mutex);
	if (g_renderer != nullptr) {
		return;
	}
	Common::InitializeThreads();
#ifdef _WIN32
	g_vulkan = LoadLibraryW(L"vulkan-1.dll");
	if (g_vulkan == nullptr) {
		Host::Fail("the Vulkan loader vulkan-1.dll is missing");
	}
	const auto get_instance_proc_addr =
	    reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(g_vulkan, "vkGetInstanceProcAddr"));
	if (get_instance_proc_addr == nullptr) {
		Host::Fail("vulkan-1.dll exports no vkGetInstanceProcAddr");
	}
#else
	g_vulkan = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
	if (g_vulkan == nullptr) {
		g_vulkan = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
	}
	if (g_vulkan == nullptr) {
		Host::Fail("the Vulkan loader libvulkan.so.1 is missing (install the Vulkan loader and a GPU driver)");
	}
	const auto get_instance_proc_addr =
	    reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(g_vulkan, "vkGetInstanceProcAddr"));
	if (get_instance_proc_addr == nullptr) {
		Host::Fail("libvulkan.so.1 exports no vkGetInstanceProcAddr");
	}
#endif
	const std::span<const char* const> extensions =
	    window != nullptr && window->extensions != nullptr
	        ? std::span<const char* const>(window->extensions, window->extensionCount)
	        : std::span<const char* const> {};
	Libs::Graphics::CreateBackendInstance(g_graphics, get_instance_proc_addr, extensions);
	const auto surface = window != nullptr ? CreateSurface(*window) : vk::SurfaceKHR {};
	Libs::Graphics::CreateBackendDevice(g_graphics, surface);
	g_renderer  = std::make_unique<Libs::Graphics::RenderContext>(g_graphics);
	g_presenter = std::make_unique<Internal::Presenter>(*g_renderer);
	if (surface != nullptr) {
		g_presenter->AttachSurface(surface, window->context);
	}
	g_video_out = std::make_unique<Libs::VideoOut::VideoOutDriver>();
	MapGuestRanges(*g_renderer);
	Libs::Graphics::ShaderRecompiler::IR::SetWalkerFastLoad(&WalkerFastLoad);
	g_published.store(g_renderer.get(), std::memory_order_release);
	GuestArena::GuestArenaSetAccessObserver_nid_postfix(&MappedThunk);
	GuestArena::GuestArenaSetReleaseObserver_nid_no_patch(&ReleasedThunk);
	GuestArena::GuestArenaSetHostWriteHook_nid_no_patch(&HostWriteThunk);
	GuestArena::GuestArenaSetHostReadHook_nid_no_patch(&HostReadThunk);
	LibcSetGpuFaultHandler_nid_no_patch(&FaultThunk);
	Libs::Graphics::ShaderInit();
	g_renderer->InitializeGpu(g_video_out.get());
	Log::WriteToConsoleAndLog("libSceAgcDriver: VRAM-resident GPU backend (KytyPS5 renderer) started\n");
}

void Core::AttachWindow(const WindowDesc& window) {
	std::lock_guard lock(g_mutex);
	if (g_renderer == nullptr) {
		Start(&window);
		return;
	}
	g_presenter->AttachSurface(CreateSurface(window), window.context);
}

void Core::Submit(std::uint32_t queue, const std::uint32_t* commands, std::uint32_t dwords) {
	if (!Started()) {
		Start(nullptr);
	}
	if (commands == nullptr || dwords == 0) {
		return;
	}
	auto& gpu = g_published.load(std::memory_order_acquire)->GetGpu();
	const std::span<const std::uint32_t> words(commands, dwords);
	// As KytyPS5's libSceAgcDriver: queues 0x20..0x57 are the compute rings, the rest graphics.
	if (queue >= 0x20 && queue < 0x58) {
		gpu.SubmitCompute(queue, words);
	} else {
		gpu.Submit(words);
	}
}

void Core::RegisterShader(const void* header) {
	// As KytyPS5's AgcCreateShader: the shader map keys programs by their code address.
	if (header == nullptr) {
		return;
	}
	if (!Started()) {
		Start(nullptr);
	}
	const auto* shader = static_cast<const Libs::Graphics::Shader*>(header);
	if (shader->code == nullptr || shader->shader_size == 0) {
		return;
	}
	Libs::Graphics::ShaderMappedData map;
	map.type                = static_cast<Libs::Graphics::Prospero::ShaderBinaryType>(shader->type);
	map.user_data           = shader->user_data;
	map.input_semantics     = shader->input_semantics;
	map.num_input_semantics = shader->num_input_semantics;
	map.code_size_bytes     = shader->shader_size;
	map.scratch_size_dwords = shader->scratch_size_dw_per_thread;
	Libs::Graphics::ShaderMapUserData(reinterpret_cast<uint64_t>(const_cast<const void*>(shader->code)), map);
}

void Core::SuspendPoint() {
	if (Started()) {
		g_published.load(std::memory_order_acquire)->GetGpu().SuspendPoint();
	}
}

void Core::WaitIdle() {
	if (Started()) {
		g_published.load(std::memory_order_acquire)->GetGpu().WaitForIdle();
	}
}

void Core::Shutdown() {
	std::lock_guard lock(g_mutex);
	if (g_renderer == nullptr) {
		return;
	}
	// Process exit: the GPU thread stops; the device, caches and presenter stay alive, since the
	// presenter thread and faulting guest threads may still be inside them.
	GuestArena::GuestArenaSetAccessObserver_nid_postfix(nullptr);
	GuestArena::GuestArenaSetReleaseObserver_nid_no_patch(nullptr);
	g_renderer->ShutdownGpu();
}

void Core::Present(const WindowDesc& window, void* key, bool hasBuffer, bool opaque,
                   void (*gpuReady)(void*), void* context) {
	if (auto* presenter = g_presenter.get(); presenter != nullptr && Started()) {
		presenter->Present(window, hasBuffer ? key : nullptr, opaque);
	}
	gpuReady(context);
}

void Core::ReleaseWindow(void* window) {
	std::lock_guard lock(g_mutex);
	if (g_presenter != nullptr) {
		g_presenter->ReleaseWindow(window);
	}
}

} // namespace AgcDriver::VramBackend

namespace Libs::VideoOut {

// AnyPS5: the flip queue, buffer reuse and vblank pacing stay in libSceVideoOut; the stack only
// reserves flips, copies their display buffers and reports when the copies completed.
namespace {
thread_local const uint32_t* g_flip_packet = nullptr;
} // namespace

struct VideoOutDriver::Impl {};

VideoOutDriver::VideoOutDriver(): m_impl(std::make_unique<Impl>()) {}

VideoOutDriver::~VideoOutDriver() = default;

int VideoOutDriver::SubmitFlipFromGpu(Graphics::CommandBuffer& buffer, int handle, int index,
                                      int flip_mode, int64_t flip_arg, uint64_t& request_id) {
	namespace Bridge = AgcDriver::VramBackend;
	Bridge::DisplayDesc display {};
	bool                has_display = false;
	if (!Bridge::Host::ReserveFlip(static_cast<uint32_t>(handle), index,
	                               static_cast<uint32_t>(flip_mode), flip_arg, g_flip_packet,
	                               &request_id, &display, &has_display)) {
		return VIDEO_OUT_ERROR_FLIP_QUEUE_FULL;
	}
	auto* presenter = Bridge::Internal::CurrentPresenter();
	auto* key       = Bridge::Host::FlipKey(request_id);
	if (has_display) {
		presenter->PrepareFrame(buffer, key, display);
	} else {
		presenter->PrepareBlank(key);
	}
	return OK;
}

void VideoOutDriver::PrepareFlip(uint64_t /*request_id*/, Graphics::CommandBuffer& /*buffer*/) {}

bool VideoOutDriver::FlipPacketReady(int handle, int index, int flip_mode, int64_t flip_arg,
                                     const uint32_t* packet) {
	return AgcDriver::VramBackend::Host::FlipPacketReady(static_cast<uint32_t>(handle), index,
	                                                     static_cast<uint32_t>(flip_mode), flip_arg,
	                                                     packet);
}

void VideoOutDriver::CompleteFlip(uint64_t request_id) {
	// Kyty counts displayed frames in its own video-out; the tracker's rewritten ("hot") pages
	// expire after a number of them (regionManager.h HotPageLifetime). Without it they never
	// expire and every page the CPU ever rewrote is uploaded again on every submission.
	Graphics::AdvanceTrackerFrame();
	// Kyty's own video-out also counts flips for the learning that compares frame rates.
	Common::FrameClock::NoteFlip();
	AgcDriver::VramBackend::Host::CompleteFlip(request_id);
}

void VideoOutDriver::WaitForSubmitSlot(int handle) {
	AgcDriver::VramBackend::Host::WaitForFlipRoom(static_cast<uint32_t>(handle));
}

void VideoOutDriver::WaitFlipDone(const uint64_t* /*label*/) {
	// AnyPS5's libSceAgc emits its own rendering-wait packet (RenderingWait below) instead.
}

bool VideoOutDriver::TakeEopFlip(uint32_t token, Graphics::CommandBuffer& buffer, uint64_t& request_id) {
	namespace Bridge = AgcDriver::VramBackend;
	Bridge::DisplayDesc display {};
	bool                has_display = false;
	if (!Bridge::Host::TakeEopFlip(token, &request_id, &display, &has_display)) {
		return false;
	}
	auto* presenter = Bridge::Internal::CurrentPresenter();
	auto* key       = Bridge::Host::FlipKey(request_id);
	if (has_display) {
		presenter->PrepareFrame(buffer, key, display);
	} else {
		presenter->PrepareBlank(key);
	}
	return true;
}

void VideoOutDriver::SetFlipPacket(const uint32_t* packet) {
	g_flip_packet = packet;
}

bool VideoOutDriver::RenderingReady(uint32_t handle, uint32_t index, const uint32_t* packet) {
	return AgcDriver::VramBackend::Host::RenderingReady(handle, index, packet);
}

} // namespace Libs::VideoOut
