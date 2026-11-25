#ifndef APS5_KYTY_SHIM_KERNEL_MEMORY_H
#define APS5_KYTY_SHIM_KERNEL_MEMORY_H

// Guest memory as the GPU stack sees it in AnyPS5. Guest addresses are host addresses: the guest
// arena lies below 1 TiB (KytyPS5's lower window) and the relinked executable's data near the top
// of the user address space falls in the "extended" window. Implemented in glue/Memory.cpp on
// top of libc's GuestArena.

#include "common/common.h"
#include "common/virtualMemory.h"

#include <cstdint>

namespace Libs::LibKernel::Memory {

constexpr uint64_t kExtendedMemoryBase = 0x7f8000000000ull;
constexpr uint64_t kExtendedMemorySize = 512ull * 1024 * 1024 * 1024;

// Copies through the direct-memory backing, ignoring the tracker's page protections; false where
// the range has no shared backing (private memory) and is not readable or writable as mapped.
bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size);
bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size);
// One byte from each of count addresses stride apart, equal to count one-byte TryReadBacking
// calls. Returns false if any address is unbacked; out is unspecified then.
bool TryReadBackingStrided(uint64_t first, uint64_t stride, uint32_t count, uint8_t* out);
// TryReadBacking after the GPU-modified bytes of the range were read back (GPU thread only).
bool TryReadBufferBacking(uint64_t vaddr, void* data, uint64_t size);
bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size);
// Reads a range that may contain unmapped pages, which read as zero.
bool CopySparseMemory(uint64_t vaddr, void* data, uint64_t size);
[[nodiscard]] uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size);
// ClampRangeSize that returns 0 for a range starting outside committed memory instead of exiting.
[[nodiscard]] uint64_t TryClampRangeSize(uint64_t vaddr, uint64_t size);
void WriteBacking(uint64_t vaddr, const void* data, uint64_t size) noexcept;
void InvalidateMemory(uint64_t vaddr, uint64_t size);
[[nodiscard]] bool ReadGpuTable(uint64_t vaddr, void* data, uint64_t size);
// AnyPS5 has no single host view of the backing: zero (the GPU-SRT host view stays off).
[[nodiscard]] uint64_t GuestBackingBase();
[[nodiscard]] uint64_t GuestBackingSize();
[[nodiscard]] bool     GuestBackingOffset(uint64_t vaddr, uint64_t* offset);
// Tracker protection (transient): the effective protection is the guest's, reduced to `mode`.
bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode);

} // namespace Libs::LibKernel::Memory

#endif
