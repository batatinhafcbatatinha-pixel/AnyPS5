// libSceAgcDriver's side of the VRAM-resident GPU backend (the KytyPS5 renderer): video outputs,
// flip reservations and EOP flip tokens, end-of-pipe interrupts and the presentation window, in
// AnyPS5's own types. The stack side is behind glue/Core.hpp.

#include "glue/Backend.hpp"
#include "glue/Core.hpp"

#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libc/include/GpuBackend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <thread>
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
	struct Prepared {
		std::uint64_t request    = 0;
		DisplayDesc   display    = {};
		bool          hasDisplay = false;
	};
	std::map<std::uint32_t, std::pair<FlipInfo, Prepared>> tokens;
	// Flips reserved at submission (on the submitting thread, as the legacy driver), by packet.
	std::map<const std::uint32_t*, std::deque<Prepared>> flips;
	// Rendering waits captured at submission, by packet address, in submission order.
	std::map<const std::uint32_t*, std::deque<std::shared_ptr<IRenderingWait>>> renderingWaits;
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

// Reserves on the calling thread (never the GPU thread: libSceVideoOut's locks may be held by a
// guest thread that waits for the GPU thread) and captures the buffer the flip shows.
Registry::Prepared ReserveNow(IVideoOutput& output, const FlipInfo& info, bool limited) {
	auto request = limited ? output.TryReserve(info) : output.Reserve(info);
	if (request == nullptr) {
		return {};
	}
	Registry::Prepared prepared;
	Describe(request, &prepared.display, &prepared.hasDisplay);
	auto&           registry = Get();
	std::lock_guard lock(registry.mutex);
	prepared.request = Remember(registry, std::move(request));
	return prepared;
}

std::shared_ptr<IVideoOutput> OutputOf(std::uint32_t handle) {
	auto&           registry = Get();
	std::lock_guard lock(registry.mutex);
	const auto      found = registry.outputs.find(handle);
	return found == registry.outputs.end() ? nullptr : found->second;
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
	// As the legacy driver: a rendering wait waits for the buffer uses known when it is submitted,
	// not for flips reserved later (an EOP flip reserved after it would deadlock it).
	const auto* words = packet->addr;
	for (std::uint32_t cursor = 0; words != nullptr && cursor < packet->dw_num;) {
		const auto header = words[cursor];
		if (header == FlipPacketHeader && cursor + FlipPacketWords <= packet->dw_num) {
			if (auto output = OutputOf(words[cursor + 1])) {
				const FlipInfo info {words[cursor + 1], static_cast<std::int32_t>(words[cursor + 2]), words[cursor + 3],
				                     static_cast<std::int64_t>(static_cast<std::uint64_t>(words[cursor + 4]) | (static_cast<std::uint64_t>(words[cursor + 5]) << 32u))};
				auto prepared = ReserveNow(*output, info, false);
				auto& registry = Get();
				std::lock_guard lock(registry.mutex);
				registry.flips[words + cursor].push_back(prepared);
			}
		}
		if (header == RenderingWaitPacketHeader && cursor + RenderingWaitPacketWords <= packet->dw_num) {
			auto& registry = Get();
			std::shared_ptr<IVideoOutput> output;
			{
				std::lock_guard lock(registry.mutex);
				if (const auto found = registry.outputs.find(words[cursor + 1]); found != registry.outputs.end()) output = found->second;
			}
			if (output != nullptr) {
				auto wait = output->CaptureRenderingWait(words[cursor + 2]);
				std::lock_guard lock(registry.mutex);
				registry.renderingWaits[words + cursor].push_back(std::move(wait));
			}
		}
		cursor += static_cast<std::uint32_t>(Pm4::PacketWords(header));
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
	if (auto output = OutputOf(info.handle)) {
		auto prepared = ReserveNow(*output, info, false);
		auto& registry = Get();
		std::lock_guard lock(registry.mutex);
		registry.flips[words].push_back(prepared);
	}
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
	auto prepared = ReserveNow(*output, info, true);
	if (prepared.request == 0) {
		return 0;
	}
	std::lock_guard lock(registry.mutex);
	registry.tokens.emplace(token, std::pair {info, prepared});
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

void RegisterShader(const void* header) {
	Core::RegisterShader(header);
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

bool FlipPacketReady(std::uint32_t handle, std::int32_t index, std::uint32_t mode, std::int64_t argument,
                     const std::uint32_t* packet) {
	auto& registry = Get();
	{
		std::lock_guard lock(registry.mutex);
		if (packet == nullptr || registry.flips.contains(packet)) {
			return true;
		}
	}
	// Not seen at submission (inside an indirect buffer): reserved here, but never by waiting for
	// libSceVideoOut's locks (a guest thread holding them may be waiting for this thread).
	auto output = OutputOf(handle);
	if (output == nullptr) {
		Fail("flip references an unregistered video output");
	}
	bool busy    = false;
	auto request = output->ReserveUnlessBusy(FlipInfo {handle, index, mode, argument}, busy);
	if (busy) {
		return false;
	}
	if (request == nullptr) {
		Fail("video output returned a null flip reservation");
	}
	Registry::Prepared prepared;
	Describe(request, &prepared.display, &prepared.hasDisplay);
	std::lock_guard lock(registry.mutex);
	prepared.request = Remember(registry, std::move(request));
	registry.flips[packet].push_back(prepared);
	return true;
}

bool ReserveFlip(std::uint32_t handle, std::int32_t index, std::uint32_t mode, std::int64_t argument,
                 const std::uint32_t* packet, std::uint64_t* request, DisplayDesc* display, bool* hasDisplay) {
	auto& registry = Get();
	{
		std::lock_guard lock(registry.mutex);
		if (const auto found = registry.flips.find(packet); packet != nullptr && found != registry.flips.end()) {
			const auto prepared = found->second.front();
			found->second.pop_front();
			if (found->second.empty()) registry.flips.erase(found);
			if (prepared.request == 0) Fail("flip reservation failed at submission");
			*request    = prepared.request;
			*display    = prepared.display;
			*hasDisplay = prepared.hasDisplay;
			return true;
		}
	}
	// A flip not seen at submission (inside an indirect buffer): reserved here.
	auto output = OutputOf(handle);
	if (output == nullptr) {
		Fail("flip references an unregistered video output");
	}
	const auto prepared = ReserveNow(*output, FlipInfo {handle, index, mode, argument}, false);
	if (prepared.request == 0) {
		Fail("video output returned a null flip reservation");
	}
	*request    = prepared.request;
	*display    = prepared.display;
	*hasDisplay = prepared.hasDisplay;
	return true;
}

bool TakeEopFlip(std::uint32_t token, std::uint64_t* request, DisplayDesc* display, bool* hasDisplay) {
	auto&           registry = Get();
	std::lock_guard lock(registry.mutex);
	const auto      found = registry.tokens.find(token);
	if (found == registry.tokens.end()) {
		return false;
	}
	*request    = found->second.second.request;
	*display    = found->second.second.display;
	*hasDisplay = found->second.second.hasDisplay;
	registry.tokens.erase(found);
	return true;
}

// GpuReady takes libSceVideoOut's locks, which a guest thread may hold while it waits for the GPU
// thread (a fault read-back); the GPU thread in turn waits for Kyty's priority operations. So the
// completion is handed to a thread of its own instead of running on the priority thread.
class FlipCompleter {
public:
	static FlipCompleter& Get() {
		static auto* completer = new FlipCompleter;
		return *completer;
	}
	void Push(std::shared_ptr<IFlipRequest> request) {
		{
			std::lock_guard lock(mutex);
			pending.push_back(std::move(request));
		}
		changed.notify_one();
	}

private:
	FlipCompleter() : worker([this] { Run(); }) { worker.detach(); }
	void Run() {
		for (;;) {
			std::shared_ptr<IFlipRequest> request;
			{
				std::unique_lock lock(mutex);
				changed.wait(lock, [this] { return !pending.empty(); });
				request = std::move(pending.front());
				pending.pop_front();
			}
			auto frame = std::make_shared<FrameTiming>(++frames);
			try {
				request->GpuReady(frame);
			} catch (...) {
				request->Fail(std::current_exception());
			}
		}
	}
	std::mutex                                 mutex;
	std::condition_variable                    changed;
	std::deque<std::shared_ptr<IFlipRequest>> pending;
	std::uint64_t                              frames = 0;
	std::thread                                worker;
};

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
	FlipCompleter::Get().Push(std::move(completed));
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

bool RenderingReady(std::uint32_t handle, std::uint32_t index, const std::uint32_t* packet) {
	auto&                           registry = Get();
	std::shared_ptr<IRenderingWait> captured;
	{
		std::lock_guard lock(registry.mutex);
		auto            found = registry.renderingWaits.find(packet);
		if (found == registry.renderingWaits.end()) {
			// Not seen at submission (inside an indirect buffer): captured now.
			std::shared_ptr<IVideoOutput> output;
			if (const auto known = registry.outputs.find(handle); known != registry.outputs.end()) output = known->second;
			if (output == nullptr) Fail("rendering wait references an unregistered video output");
			registry.mutex.unlock();
			auto wait = output->CaptureRenderingWait(index);
			registry.mutex.lock();
			found = registry.renderingWaits.find(packet);
			if (found == registry.renderingWaits.end()) found = registry.renderingWaits.emplace(packet, std::deque<std::shared_ptr<IRenderingWait>> {}).first;
			found->second.push_front(std::move(wait));
		}
		captured = found->second.front();
	}
	if (captured != nullptr && !captured->Ready()) {
		return false;
	}
	std::lock_guard lock(registry.mutex);
	if (const auto found = registry.renderingWaits.find(packet); found != registry.renderingWaits.end()) {
		found->second.pop_front();
		if (found->second.empty()) registry.renderingWaits.erase(found);
	}
	return true;
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
