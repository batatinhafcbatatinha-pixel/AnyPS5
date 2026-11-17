#ifndef APS5_KYTY_SHIM_KERNEL_PTHREAD_H
#define APS5_KYTY_SHIM_KERNEL_PTHREAD_H

#include <cstdint>

namespace Libs::LibKernel {

// Host time stamp counter (QueryPerformanceCounter on Windows) and its frequency.
uint64_t KernelGetTscFrequency();
uint64_t KernelReadTsc();

} // namespace Libs::LibKernel

#endif
