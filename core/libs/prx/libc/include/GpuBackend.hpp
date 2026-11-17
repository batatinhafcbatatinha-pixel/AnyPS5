#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GPUBACKEND_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GPUBACKEND_HPP

#include <cstdlib>
#include <cstring>

namespace GpuBackend {

// The GPU driver backend: the VRAM-resident one (default on Windows) or the legacy host-import
// one (APS5_GPU_BACKEND=legacy, and every other platform). libc, libkernel and libSceAgcDriver
// must agree, so the choice is read from the environment here.
inline bool VramResident() {
#ifdef _WIN32
    static const bool selected = [] {
        const char* value = std::getenv("APS5_GPU_BACKEND");
        return value == nullptr || std::strcmp(value, "legacy") != 0;
    }();
    return selected;
#else
    return false;
#endif
}

}

#endif
