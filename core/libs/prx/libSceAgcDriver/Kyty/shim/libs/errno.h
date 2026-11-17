#ifndef APS5_KYTY_SHIM_LIBS_ERRNO_H
#define APS5_KYTY_SHIM_LIBS_ERRNO_H

namespace Libs {

constexpr int OK = 0;

namespace LibKernel {
constexpr int KERNEL_ERROR_ENOENT = static_cast<int>(0x80020002u);
constexpr int KERNEL_ERROR_EBADF  = static_cast<int>(0x80020009u);
} // namespace LibKernel

namespace VideoOut {
constexpr int VIDEO_OUT_ERROR_FLIP_QUEUE_FULL = static_cast<int>(0x80290012u);
} // namespace VideoOut

} // namespace Libs

#endif
