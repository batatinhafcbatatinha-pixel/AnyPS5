// libSceAgcDriver's side of the VRAM-resident GPU backend (the KytyPS5 renderer): video outputs,
// flip reservations and EOP flip tokens, end-of-pipe interrupts and the presentation window, in
// AnyPS5's own types. The stack side is behind glue/Core.hpp.

#include "glue/Backend.hpp"
#include "glue/Core.hpp"

#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libc/include/GpuBackend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace AgcDriver::VramBackend {

namespace {

struct Registry {
	std::mutex                                                  mutex;
	std::map<std::uint32_t, std::shared_ptr<IVideoOutput>>     outputs;
	std::map<std::uint64_t, std::shared_ptr<IFlipRequest>>     requests;
	std::map<std::uint32_t, std::pair<FlipInfo, std::uint64_t>> tokens;
	std::uint64_t                                               nextRequest = 1;
	std::uint32_t                                               nextToken   = EopFlipTokenBase;
	std::atomic<std::uint64_t>                                  frames {0};
};

Registry& Get() {
	static auto* registry = new Registry;
	return *registry;
}

void Describe(const std::shared_ptr<IFlipRequest>& request, DisplayDesc* display, bool* hasDisplay) {
	DisplayBuffer buffer {};
	*hasDisplay = request->Display(buffer);
	if (!*hasDisplay) {
		return;
	}
	display->address       = buffer.address;
	display->pixelFormat   = buffer.pixelFormat;
	display->width         = buffer.width;
	display->height        = buffer.height;
	display->tilingMode    = buffer.tilingMode;
	display->pitchInPixel  = buffer.pitchInPixel;
	display->dccAddress    = buffer.dccAddress;
	display->dccClearColor = buffer.dccClearColor;
	display->dccControl    = buffer.dccControl;
}

std::uint64_t Remember(Registry& registry, std::shared_ptr<IFlipRequest> request) {
	const auto id = registry.nextRequest++;
	registry.requests.emplace(id, std::move(request));
	return id;
}

// The window's surface callback takes a VkInstance; the stack passes it as a pointer.
thread_local const PresentationWindow* g_window = nullptr;

std::uint64_t CreateSurfaceThunk(void* context, void* instance) {
	const auto* window = g_window;
	if (window == nullptr || window->createSurface == nullptr || window->context != context) {
		return 0;
	}
	return reinterpret_cast<std::uint64_t>(window->createSurface(context, static_cast<VkInstance>(instance)));
}

WindowDesc WindowFor(const PresentationWindow& window) {
	WindowDesc desc;
	desc.context         = window.context;
	desc.extensions      = window.extensions.data();
	desc.extensionCount  = window.extensions.size();
	desc.createSurface   = &CreateSurfaceThunk;
	desc.getDrawableSize = window.getDrawableSize;
	return desc;
}

} // namespace

bool Enabled() {
	return GpuBackend::VramResident();
}

void Submit(const Packet* packet, std::uint32_t queue) {
	if (packet == nullptr) {
		throw std::runtime_error("AGC: null command buffer submission");
	}
	if (packet->flags != 0) {
		throw std::runtime_error("AGC: nonzero submission flags are not implemented");
	}
	Core::Submit(queue, packet->addr, packet->dw_num);
}

void SubmitFlip(const FlipInfo& info) {
	// AnyPS5's flip packet, kept alive until the GPU thread has read it (it is consumed in order;
	// a small ring is enough because each flip is reserved before the next is accepted).
	static std::mutex                                   mutex;
	static std::vector<std::array<std::uint32_t, FlipPacketWords>> ring(64);
	static std::size_t                                  next = 0;
	std::uint32_t*                                      words = nullptr;
	{
		std::lock_guard lock(mutex);
		words = ring[next].data();
		next  = (next + 1) % ring.size();
	}
	const auto argument = static_cast<std::uint64_t>(info.argument);
	words[0]            = FlipPacketHeader;
	words[1]            = info.handle;
	words[2]            = static_cast<std::uint32_t>(info.index);
	words[3]            = info.mode;
	words[4]            = static_cast<std::uint32_t>(argument);
	words[5]            = static_cast<std::uint32_t>(argument >> 32u);
	Core::Submit(0, words, FlipPacketWords);
}

std::uint32_t ReserveEopFlip(const FlipInfo& info, std::uint32_t token) {
	auto& registry = Get();
	std::shared_ptr<IVideoOutput> output;
	{
		std::lock_guard lock(registry.mutex);
		const auto found = registry.outputs.find(info.handle);
		if (found == registry.outputs.end()) {
			throw std::runtime_error("EOP flip references an unregistered video output");
		}
		output = found->second;
		if (token == 0) {
			token = registry.nextToken;
		}
		if (token < EopFlipTokenBase || token > EopFlipTokenLimit) {
			throw std::runtime_error("EOP flip token is out of range");
		}
		if (const auto reserved = registry.tokens.find(token); reserved != registry.tokens.end()) {
			const auto& known = reserved->second.first;
			if (known.handle != info.handle || known.index != info.index || known.mode != info.mode ||
			    known.argument != info.argument) {
				throw std::runtime_error("EOP flip token is reserved for a different flip");
			}
			throw EopFlipAlreadyReserved {};
		}
	}
	auto request = output->TryReserve(info);
	if (!request) {
		return 0;
	}
	std::lock_guard lock(registry.mutex);
	registry.tokens.emplace(token, std::pair {info, Remember(registry, std::move(request))});
	registry.nextToken = std::max(registry.nextToken, token + 1);
	if (registry.nextToken > EopFlipTokenLimit) {
		registry.nextToken = EopFlipTokenBase;
	}
	return token;
}

void RegisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output) {
	if (output == nullptr) {
		throw std::runtime_error("null video output");
	}
	auto&           registry = Get();
	std::lock_guard lock(registry.mutex);
	if (!registry.outputs.emplace(handle, output).second) {
		throw std::runtime_error("video output already registered");
	}
}

void UnregisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output) {
	auto&           registry = Get();
	std::lock_guard lock(registry.mutex);
	const auto      found = registry.outputs.find(handle);
	if (found == registry.outputs.end() || found->second != output) {
		throw std::runtime_error("video output registration mismatch");
	}
	std::erase_if(registry.tokens, [handle](const auto& entry) { return entry.second.first.handle == handle; });
	registry.outputs.erase(found);
}

void AttachWindow(const PresentationWindow& window) {
	g_window = &window;
	const auto desc = WindowFor(window);
	Core::AttachWindow(desc);
	g_window = nullptr;
}

void Present(const PresentationWindow& window, const DisplayBuffer* buffer, const DisplayBuffer* /*plane*/,
             bool opaque, void (*gpuReady)(void*), void* context) {
	if (gpuReady == nullptr || context == nullptr) {
		throw std::runtime_error("missing GPU completion callback");
	}
	g_window        = &window;
	const auto desc = WindowFor(window);
	Core::Present(desc, context, buffer != nullptr, opaque, gpuReady, context);
	g_window = nullptr;
}

void ReleaseWindow(void* window) {
	Core::ReleaseWindow(window);
}

void SuspendPoint() {
	Core::SuspendPoint();
}

void WaitIdle() {
	Core::WaitIdle();
}

void Shutdown() {
	Core::Shutdown();
}

namespace Host {

bool ReserveFlip(std::uint32_t handle, std::int32_t index, std::uint32_t mode, std::int64_t argument,
                 std::uint64_t* request, DisplayDesc* display, bool* hasDisplay) {
	auto& registry = Get();
	std::shared_ptr<IVideoOutput> output;
	{
		std::lock_guard lock(registry.mutex);
		const auto found = registry.outputs.find(handle);
		if (found == registry.outputs.end()) {
			Fail("flip references an unregistered video output");
		}
		output = found->second;
	}
	auto reserved = output->Reserve(FlipInfo {handle, index, mode, argument});
	if (reserved == nullptr) {
		Fail("video output returned a null flip reservation");
	}
	Describe(reserved, display, hasDisplay);
	std::lock_guard lock(registry.mutex);
	*request = Remember(registry, std::move(reserved));
	return true;
}

bool TakeEopFlip(std::uint32_t token, std::uint64_t* request, DisplayDesc* display, bool* hasDisplay) {
	auto&                         registry = Get();
	std::shared_ptr<IFlipRequest> reserved;
	{
		std::lock_guard lock(registry.mutex);
		const auto      found = registry.tokens.find(token);
		if (found == registry.tokens.end()) {
			return false;
		}
		*request = found->second.second;
		registry.tokens.erase(found);
		reserved = registry.requests.at(*request);
	}
	Describe(reserved, display, hasDisplay);
	return true;
}

void CompleteFlip(std::uint64_t request) {
	auto&                         registry = Get();
	std::shared_ptr<IFlipRequest> completed;
	{
		std::lock_guard lock(registry.mutex);
		const auto      found = registry.requests.find(request);
		if (found == registry.requests.end()) {
			return;
		}
		completed = std::move(found->second);
		registry.requests.erase(found);
	}
	auto frame = std::make_shared<FrameTiming>(registry.frames.fetch_add(1, std::memory_order_relaxed) + 1);
	try {
		completed->GpuReady(frame);
	} catch (...) {
		completed->Fail(std::current_exception());
	}
}

void* FlipKey(std::uint64_t request) {
	auto&           registry = Get();
	std::lock_guard lock(registry.mutex);
	const auto      found = registry.requests.find(request);
	return found == registry.requests.end() ? nullptr : dynamic_cast<void*>(found->second.get());
}

void WaitForFlipRoom(std::uint32_t handle) {
	std::shared_ptr<IVideoOutput> output;
	{
		auto&           registry = Get();
		std::lock_guard lock(registry.mutex);
		if (const auto found = registry.outputs.find(handle); found != registry.outputs.end()) {
			output = found->second;
		}
	}
	if (output != nullptr) {
		output->WaitForFlipRoom();
	}
}

void RenderingWait(std::uint32_t handle, std::uint32_t index) {
	std::shared_ptr<IVideoOutput> output;
	{
		auto&           registry = Get();
		std::lock_guard lock(registry.mutex);
		if (const auto found = registry.outputs.find(handle); found != registry.outputs.end()) {
			output = found->second;
		}
	}
	if (output == nullptr) {
		Fail("rendering wait references an unregistered video output");
	}
	if (auto wait = output->CaptureRenderingWait(index)) {
		wait->Wait();
	}
}

void DeliverInterrupt(std::uint32_t queue, std::uint32_t context) {
	AgcDriverDeliverEopInterrupt(queue, context);
}

void Fail(const char* message) {
	std::fprintf(stderr, "[agc] VRAM-resident GPU backend: %s\n", message);
	std::fflush(stderr);
	std::abort();
}

} // namespace Host

} // namespace AgcDriver::VramBackend
