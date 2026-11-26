#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GPUBACKEND_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GPUBACKEND_HPP

#include <cstdlib>
#include <cstring>

namespace GpuBackend {

// The GPU driver backend: the VRAM-resident one (default on Windows) or the legacy host-import
// one (APS5_GPU_BACKEND=legacy, and every other platform). libc, libkernel and libSceAgcDriver
// must agree, so the choice is read from the environment here.
// Linux: the legacy driver is the default; a build with ANYPS5_KYTY_GPU_LINUX=ON selects the
// VRAM-resident one with APS5_GPU_BACKEND=vram.
inline bool VramResident() {
#ifdef _WIN32
    static const bool selected = [] {
        const char* value = std::getenv("APS5_GPU_BACKEND");
        return value == nullptr || std::strcmp(value, "legacy") != 0;
    }();
    return selected;
#elif defined(__linux__) && defined(APS5_KYTY_LINUX)
    static const bool selected = [] {
        const char* value = std::getenv("APS5_GPU_BACKEND");
        return value != nullptr && std::strcmp(value, "vram") == 0;
    }();
    return selected;
#else
    return false;
#endif
}

}

#endif
