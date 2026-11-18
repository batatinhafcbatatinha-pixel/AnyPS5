#ifndef APS5_KYTY_GLUE_CORE_HPP
#define APS5_KYTY_GLUE_CORE_HPP

// The seam between AnyPS5's libSceAgcDriver (Backend.cpp, AnyPS5 types only) and the KytyPS5 GPU
// stack (Core.cpp, Memory.cpp, Presenter.cpp, Kyty types only). Only plain types cross it.

#include <cstddef>
#include <cstdint>

namespace AgcDriver::VramBackend {

struct DisplayDesc {
	std::uint64_t address       = 0;
	std::uint64_t pixelFormat   = 0;
	std::uint32_t width         = 0;
	std::uint32_t height        = 0;
	std::uint32_t tilingMode    = 0;
	std::uint32_t pitchInPixel  = 0;
	std::uint64_t dccAddress    = 0;
	std::uint64_t dccClearColor = 0;
	std::uint32_t dccControl    = 0;
};

struct WindowDesc {
	void*              context = nullptr;
	const char* const* extensions = nullptr;
	std::size_t        extensionCount = 0;
	// Returns a VkSurfaceKHR for the instance (both as 64-bit handles).
	std::uint64_t (*createSurface)(void* context, void* instance) = nullptr;
	void (*getDrawableSize)(void* context, std::uint32_t* width, std::uint32_t* height) = nullptr;
};

// Implemented by Backend.cpp (AnyPS5 side), called by the stack.
namespace Host {

// Reserves the flip with libSceVideoOut; false when the output is not registered. `display` gets
// the buffer it shows and `hasDisplay` whether it shows one (false: black or blank).
// `packet`: the flip packet being executed (its flip was reserved when it was submitted), or null.
bool ReserveFlip(std::uint32_t handle, std::int32_t index, std::uint32_t mode, std::int64_t argument,
                 const std::uint32_t* packet, std::uint64_t* request, DisplayDesc* display, bool* hasDisplay);
// GPU thread, before a flip packet: true when its flip is reserved (at submission, or now without
// waiting); false when libSceVideoOut's locks are held: the packet waits and is tried again.
bool FlipPacketReady(std::uint32_t handle, std::int32_t index, std::uint32_t mode, std::int64_t argument,
                     const std::uint32_t* packet);
// An end-of-pipe release whose interrupt context is a reserved EOP flip token (queue 0).
bool TakeEopFlip(std::uint32_t token, std::uint64_t* request, DisplayDesc* display, bool* hasDisplay);
// The work before the flip completed on the GPU: libSceVideoOut may present it.
void CompleteFlip(std::uint64_t request);
// The address libSceVideoOut passes back to Present for this request.
void* FlipKey(std::uint64_t request);
void WaitForFlipRoom(std::uint32_t handle);
// `packet`: the wait packet in guest memory; its wait was captured when it was submitted.
bool RenderingReady(std::uint32_t handle, std::uint32_t index, const std::uint32_t* packet);
void DeliverInterrupt(std::uint32_t queue, std::uint32_t context);
// GPU-backend fatal error: report it through libSceAgcDriver's failure path.
[[noreturn]] void Fail(const char* message);

} // namespace Host

// Implemented by the stack side.
namespace Core {

bool Started();
// Device (and window surface, when given) creation and the GPU thread; idempotent.
void Start(const WindowDesc* window);
void AttachWindow(const WindowDesc& window);
void Submit(std::uint32_t queue, const std::uint32_t* commands, std::uint32_t dwords);
void SuspendPoint();
// sceAgcCreateShader: the relocated shader header (SDK layout) of a program the game created.
void RegisterShader(const void* header);
void WaitIdle();
void Shutdown();
// Presenter thread: shows the frame recorded for `key` (or a clear) and calls gpuReady(context).
void Present(const WindowDesc& window, void* key, bool hasBuffer, bool opaque, void (*gpuReady)(void*), void* context);
void ReleaseWindow(void* window);
// Guest memory notifications (libc observers).
void NoteMapped(std::uint64_t address, std::size_t bytes);
void NoteReleased(std::uint64_t address, std::size_t bytes);
void NoteHostWrite(std::uint64_t address, std::size_t bytes);
// The host is about to read the range (file writes, sockets): GPU-written bytes come back first.
void NoteHostRead(std::uint64_t address, std::size_t bytes);
// Vectored-exception hook (EXCEPTION_POINTERS*): true when the fault was the tracker's.
bool HandleFault(void* exceptionPointers) noexcept;

} // namespace Core

} // namespace AgcDriver::VramBackend

#endif
