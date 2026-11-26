#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_HOSTFAULT_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_HOSTFAULT_HPP

#include <cstdint>

#ifdef _WIN32
using LibcHostFaultHandler = bool (*)(void* exceptionPointers) noexcept;

extern "C" void LibcSetHostFaultHandler_nid_no_patch(LibcHostFaultHandler handler) noexcept;
// Access violations on guest pages the GPU backend protects; runs before every other handler.
using LibcGpuFaultHandler = bool (*)(void* exceptionPointers) noexcept;
extern "C" void LibcSetGpuFaultHandler_nid_no_patch(LibcGpuFaultHandler handler) noexcept;
#elif defined(__linux__)
// Page faults (SIGSEGV) on guest pages the GPU backend protects: siginfo_t* and ucontext_t* of the
// signal. Called on the faulting thread before the crash report; true when the access can be
// retried.
using LibcGpuFaultHandler = bool (*)(void* signalInfo, void* signalContext) noexcept;
extern "C" void LibcSetGpuFaultHandler_nid_no_patch(LibcGpuFaultHandler handler) noexcept;
#endif

#endif
