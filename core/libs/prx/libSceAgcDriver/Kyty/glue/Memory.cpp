// Guest memory services for the KytyPS5 GPU stack inside AnyPS5 (kernel/memory.h of the shim).
//
// Guest addresses are host addresses. The stack's page tracker (PageManager) write-protects pages
// whose bytes a cache uploaded and read-protects pages whose newest bytes only the GPU holds; the
// protections here are "transient": each reduced page remembers the protection the guest gave it
// and gets it back when the tracker lets go. Faults on reduced pages reach the stack through
// libc's GPU fault slot (Core::HandleFault).
//
// The SRT-walker fault path follows KytyPS5 src/loader/runtimeLinker.cpp (KytyExceptionHandler) and
// src/kernel/memory.cpp (TryReadBufferBacking, ReadGpuTable, WalkerFastLoad), commit 3a7d8030,
// GPL-2.0.

#include "glue/Core.hpp"
#include "glue/CoreInternal.hpp"

#include "common/assert.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "loader/systemContent.h"

#include "prx/libc/include/GuestArena.hpp"
#include "prx/libkernel/AppMetadata/include/AppMetadata.hpp"

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace AgcDriver::VramBackend {

namespace {

constexpr uint64_t PageBytes = 0x1000;

std::mutex& ProtectionMutex() {
	static std::mutex mutex;
	return mutex;
}

// Page -> the protection the guest gave it, for every page the tracker reduced.
std::unordered_map<uint64_t, DWORD>& Originals() {
	static std::unordered_map<uint64_t, DWORD> originals;
	return originals;
}

DWORD Reduce(DWORD original, Common::VirtualMemory::Mode mode) {
	if (mode == Common::VirtualMemory::Mode::NoAccess) {
		return PAGE_NOACCESS;
	}
	const DWORD modifiers = original & (PAGE_GUARD | PAGE_NOCACHE | PAGE_WRITECOMBINE);
	switch (original & 0xffu) {
		case PAGE_READWRITE:
		case PAGE_WRITECOPY: return PAGE_READONLY | modifiers;
		case PAGE_EXECUTE_READWRITE:
		case PAGE_EXECUTE_WRITECOPY: return PAGE_EXECUTE_READ | modifiers;
		default: return original;
	}
}

bool Protect(uint64_t address, uint64_t bytes, DWORD protection) {
	DWORD previous = 0;
	return VirtualProtect(reinterpret_cast<void*>(address), bytes, protection, &previous) != 0;
}

// Applies `mode` to [begin, end), page by page in runs of equal protection. Mutex held.
void ApplyLocked(uint64_t begin, uint64_t end, Common::VirtualMemory::Mode mode) {
	auto& originals = Originals();
	uint64_t address = begin;
	while (address < end) {
		MEMORY_BASIC_INFORMATION memory {};
		if (VirtualQuery(reinterpret_cast<void*>(address), &memory, sizeof(memory)) == 0) {
			return;
		}
		const auto region_end =
		    std::min(end, reinterpret_cast<uint64_t>(memory.BaseAddress) + memory.RegionSize);
		if (memory.State != MEM_COMMIT) {
			for (uint64_t page = address; page < region_end; page += PageBytes) {
				originals.erase(page);
			}
			address = region_end;
			continue;
		}
		uint64_t run_begin  = address;
		DWORD    run_target = 0;
		bool     run_open   = false;
		const auto flush    = [&](uint64_t run_end) {
			if (run_open && run_end > run_begin) {
				(void)Protect(run_begin, run_end - run_begin, run_target);
			}
			run_open = false;
		};
		for (uint64_t page = address; page < region_end; page += PageBytes) {
			DWORD target = 0;
			if (mode == Common::VirtualMemory::Mode::ReadWrite) {
				const auto found = originals.find(page);
				if (found == originals.end()) {
					flush(page);
					continue;
				}
				target = found->second;
				originals.erase(found);
			} else {
				const auto [found, inserted] = originals.try_emplace(page, memory.Protect);
				target                       = Reduce(found->second, mode);
			}
			if (!run_open || target != run_target) {
				flush(page);
				run_begin  = page;
				run_target = target;
				run_open   = true;
			}
		}
		flush(region_end);
		address = region_end;
	}
}

// Set while a backing read runs a guarded copy: its faults belong to libc's read recovery.
thread_local bool t_guarded_read = false;

// A plain copy of guest bytes that fails instead of faulting (no tracker read-back, no page walk).
// APS5_KYTY_MEMSTATS=1: guarded reads (calls, bytes, failures) and slow reads, every 10 s.
struct MemStats {
	std::atomic<uint64_t> calls {0}, bytes {0}, failed {0}, slow {0}, slow_bytes {0};
	std::atomic<int64_t>  next {0};
};
MemStats& Stats() {
	static MemStats stats;
	return stats;
}
void ReportStats() {
	static const bool enabled = std::getenv("APS5_KYTY_MEMSTATS") != nullptr;
	if (!enabled) {
		return;
	}
	auto&      stats = Stats();
	const auto now   = std::chrono::steady_clock::now().time_since_epoch().count();
	auto       next  = stats.next.load(std::memory_order_relaxed);
	if (now < next || !stats.next.compare_exchange_strong(next, now + 10'000'000'000ll)) {
		return;
	}
	if (next == 0) {
		return;
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "[memstats] 10 s: guarded reads {} ({} MiB), failed {}, slow reads {} ({} MiB)",
	    stats.calls.exchange(0), stats.bytes.exchange(0) >> 20, stats.failed.exchange(0),
	    stats.slow.exchange(0), stats.slow_bytes.exchange(0) >> 20) + std::string(1, char(10)));
}

bool GuardedRead(uint64_t address, void* data, uint64_t bytes) {
	t_guarded_read = true;
	const bool copied = GuestArena::GuestArenaCopyOwned_nid_no_patch(address, data, bytes);
	t_guarded_read = false;
	auto& stats = Stats();
	stats.calls.fetch_add(1, std::memory_order_relaxed);
	stats.bytes.fetch_add(bytes, std::memory_order_relaxed);
	if (!copied) {
		stats.failed.fetch_add(1, std::memory_order_relaxed);
	}
	ReportStats();
	return copied;
}

bool Tracked(uint64_t address) {
	std::lock_guard lock(ProtectionMutex());
	return Originals().contains(address & ~(PageBytes - 1));
}

bool Readable(uint64_t address, uint64_t bytes) {
	const auto end = address + bytes;
	while (address < end) {
		MEMORY_BASIC_INFORMATION memory {};
		if (VirtualQuery(reinterpret_cast<void*>(address), &memory, sizeof(memory)) == 0 ||
		    memory.State != MEM_COMMIT || (memory.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
			return false;
		}
		address = reinterpret_cast<uint64_t>(memory.BaseAddress) + memory.RegionSize;
	}
	return true;
}

// Copies with the pages opened to the guest's protection for the duration (private memory has
// no shared backing to go through). A guest store racing on these pages escapes the tracker.
bool CopyOpened(uint64_t address, void* data, uint64_t bytes, bool write) {
	std::lock_guard lock(ProtectionMutex());
	auto&           originals = Originals();
	const auto      begin     = address & ~(PageBytes - 1);
	const auto      end       = (address + bytes + PageBytes - 1) & ~(PageBytes - 1);
	std::vector<std::pair<uint64_t, DWORD>> reduced;
	const auto restore = [&] {
		for (const auto& [opened, protection]: reduced) {
			(void)Protect(opened, PageBytes, protection);
		}
	};
	for (uint64_t page = begin; page < end; page += PageBytes) {
		MEMORY_BASIC_INFORMATION memory {};
		if (VirtualQuery(reinterpret_cast<void*>(page), &memory, sizeof(memory)) == 0 ||
		    memory.State != MEM_COMMIT) {
			restore();
			return false;
		}
		if (const auto found = originals.find(page); found != originals.end()) {
			if (!Protect(page, PageBytes, write ? PAGE_READWRITE : found->second)) {
				restore();
				return false;
			}
			reduced.emplace_back(page, memory.Protect);
		} else if (write ? (memory.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY |
		                                      PAGE_EXECUTE_WRITECOPY)) == 0
		                 : (memory.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
			restore();
			return false;
		}
	}
	// `data` is host memory (staging): a guest page there would fault into this mutex.
	if (write) {
		std::memcpy(reinterpret_cast<void*>(address), data, bytes);
	} else {
		std::memcpy(data, reinterpret_cast<const void*>(address), bytes);
	}
	restore();
	return true;
}

} // namespace

bool Core::HandleFault(void* exception_pointers) noexcept {
	auto* const info   = static_cast<EXCEPTION_POINTERS*>(exception_pointers);
	const auto* record = info->ExceptionRecord;
	auto*       context = info->ContextRecord;
	auto*       renderer = Internal::Renderer();
	if (renderer == nullptr || t_guarded_read) {
		return false;
	}
	const auto kind    = record->ExceptionInformation[0];
	const auto address = static_cast<uint64_t>(record->ExceptionInformation[1]);
	namespace IR       = Libs::Graphics::ShaderRecompiler::IR;
	const auto skip_load = [context](uint32_t length, uint32_t value) {
		context->Rax = value;
		context->Rip += length;
	};
	if (kind == 0) {
		const auto length = IR::SrtWalkerLoadLength(context->Rip);
		if (length != 0 && IR::WalkerSkipsGpuWritten()) {
			skip_load(length, 0);
			IR::MarkWalkerTainted();
			return true;
		}
		uint32_t value = 0;
		if (length != 0) {
			IR::SrtWalkerNoteProtectedPage(context->R10);
		}
		if (length != 0 && Libs::Graphics::GuestGpu::IsGpuThread() &&
		    renderer->GetBufferCache().TryCurrentWalkerLoad(context->R10, value)) {
			skip_load(length, value);
			return true;
		}
	}
	if (!Tracked(address)) {
		return false;
	}
	const auto access = kind == 1   ? Libs::Graphics::PageFaultAccess::Write
	                    : kind == 8 ? Libs::Graphics::PageFaultAccess::Execute
	                                : Libs::Graphics::PageFaultAccess::Read;
	if (renderer->HandleFault(access, address)) {
		return true;
	}
	{
		// A reduced page the renderer does not consider mapped: give the guest its protection
		// back rather than crash, and report it.
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
			Log::WriteToConsoleAndLog(fmt::format("[fault] tracker page 0x{:x} ({}) refused by the renderer (mapped {}); protection restored\n",
			                                      address, kind == 1 ? "write" : "read", renderer->IsMapped(address, 1) ? 1 : 0));
		}
		std::lock_guard lock(ProtectionMutex());
		const auto page = address & ~(PageBytes - 1);
		if (const auto found = Originals().find(page); found != Originals().end()) {
			(void)Protect(page, PageBytes, found->second);
			Originals().erase(found);
			return true;
		}
	}
	if (const auto length = IR::SrtWalkerLoadLength(context->Rip); length != 0) {
		skip_load(length, 0);
		return true;
	}
	return false;
}

void Core::NoteMapped(std::uint64_t address, std::size_t bytes) {
	auto* renderer = Internal::Renderer();
	if (renderer == nullptr || bytes == 0) {
		return;
	}
	{
		// A guest protection change replaced the tracker's on these pages: remember the new
		// guest protection and let the tracker reduce them again.
		std::lock_guard lock(ProtectionMutex());
		auto&           originals = Originals();
		for (uint64_t page = address & ~(PageBytes - 1); page < address + bytes; page += PageBytes) {
			originals.erase(page);
		}
	}
	if (Libs::Graphics::GuestRange {address, bytes}.Valid()) {
		renderer->MapMemory(address, bytes);
		(void)renderer->InvalidateMemory(address, bytes);
	}
}

void Core::NoteReleased(std::uint64_t address, std::size_t bytes) {
	auto* renderer = Internal::Renderer();
	if (renderer == nullptr || bytes == 0 || !Libs::Graphics::GuestRange {address, bytes}.Valid()) {
		return;
	}
	renderer->UnmapMemory(address, bytes);
	std::lock_guard lock(ProtectionMutex());
	auto&           originals = Originals();
	for (uint64_t page = address & ~(PageBytes - 1); page < address + bytes; page += PageBytes) {
		originals.erase(page);
	}
}

void Core::NoteHostRead(std::uint64_t address, std::size_t bytes) {
	auto* renderer = Internal::Renderer();
	if (renderer == nullptr || bytes == 0 || !Libs::Graphics::GuestRange {address, bytes}.Valid()) {
		return;
	}
	bool reduced = false;
	{
		std::lock_guard lock(ProtectionMutex());
		for (uint64_t page = address & ~(PageBytes - 1); page < address + bytes && !reduced; page += PageBytes) {
			reduced = Originals().contains(page);
		}
	}
	if (reduced) {
		renderer->GetBufferCache().ReadMemory(address, bytes);
	}
}

void Core::NoteHostWrite(std::uint64_t address, std::size_t bytes) {
	if (bytes != 0) {
		Libs::LibKernel::Memory::InvalidateMemory(address, bytes);
	}
}

} // namespace AgcDriver::VramBackend

namespace Libs::LibKernel::Memory {

using AgcDriver::VramBackend::Internal::Renderer;

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) {
	if (size == 0) {
		return true;
	}
	std::lock_guard lock(AgcDriver::VramBackend::ProtectionMutex());
	AgcDriver::VramBackend::ApplyLocked(vaddr & ~(AgcDriver::VramBackend::PageBytes - 1),
	                                    (vaddr + size + AgcDriver::VramBackend::PageBytes - 1) &
	                                        ~(AgcDriver::VramBackend::PageBytes - 1),
	                                    mode);
	return true;
}

bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
	if (size == 0) {
		return true;
	}
	if (GuestArena::GuestArenaBackingCopy_nid_no_patch(vaddr, const_cast<void*>(data), size, true)) {
		return true;
	}
	return AgcDriver::VramBackend::CopyOpened(vaddr, const_cast<void*>(data), size, true);
}

bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
	if (size == 0) {
		return true;
	}
	// Fast path (most reads): the pages are readable as mapped.
	if (AgcDriver::VramBackend::GuardedRead(vaddr, data, size)) {
		return true;
	}
	AgcDriver::VramBackend::Stats().slow.fetch_add(1, std::memory_order_relaxed);
	AgcDriver::VramBackend::Stats().slow_bytes.fetch_add(size, std::memory_order_relaxed);
	if (GuestArena::GuestArenaBackingCopy_nid_no_patch(vaddr, data, size, false)) {
		return true;
	}
	if (AgcDriver::VramBackend::Readable(vaddr, size)) {
		std::memcpy(data, reinterpret_cast<const void*>(vaddr), size);
		return true;
	}
	return AgcDriver::VramBackend::CopyOpened(vaddr, data, size, false);
}

bool TryReadBufferBacking(uint64_t vaddr, void* data, uint64_t size) {
	auto* renderer = Renderer();
	if (renderer != nullptr && Graphics::GuestRange {vaddr, size}.Valid()) {
		if (!Graphics::GuestGpu::IsGpuThread()) {
			return false;
		}
		auto& buffers = renderer->GetBufferCache();
		if (buffers.HasGpuDirtyBytes(vaddr, size)) {
			buffers.ReadMemory(vaddr, size);
		}
	}
	return TryReadBacking(vaddr, data, size);
}

static bool CopySparse(uint64_t vaddr, void* data, uint64_t size, bool zero_unmapped) {
	if (AgcDriver::VramBackend::GuardedRead(vaddr, data, size)) {
		return true;
	}
	auto*      out = static_cast<std::byte*>(data);
	const auto end = vaddr + size;
	while (vaddr < end) {
		MEMORY_BASIC_INFORMATION memory {};
		if (VirtualQuery(reinterpret_cast<void*>(vaddr), &memory, sizeof(memory)) == 0) {
			return false;
		}
		const auto region_end =
		    std::min(end, reinterpret_cast<uint64_t>(memory.BaseAddress) + memory.RegionSize);
		const auto bytes = region_end - vaddr;
		// As KytyPS5: gaps (reserved, free or inaccessible guest pages) read as zero.
		static_cast<void>(zero_unmapped);
		if (memory.State != MEM_COMMIT || !TryReadBacking(vaddr, out, bytes)) {
			std::memset(out, 0, bytes);
		}
		out += bytes;
		vaddr = region_end;
	}
	return true;
}

bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size) {
	return CopySparse(vaddr, data, size, false);
}

bool CopySparseMemory(uint64_t vaddr, void* data, uint64_t size) {
	return CopySparse(vaddr, data, size, true);
}

uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size) {
	uint64_t       committed = 0;
	const uint64_t end       = vaddr + size;
	uint64_t       address   = vaddr;
	while (address < end) {
		MEMORY_BASIC_INFORMATION memory {};
		if (VirtualQuery(reinterpret_cast<void*>(address), &memory, sizeof(memory)) == 0 ||
		    memory.State != MEM_COMMIT) {
			break;
		}
		const auto region_end =
		    std::min(end, reinterpret_cast<uint64_t>(memory.BaseAddress) + memory.RegionSize);
		committed += region_end - address;
		address = region_end;
	}
	if (committed == 0) {
		EXIT("Memory: attempted to access invalid address 0x%016" PRIx64 " with size 0x%016" PRIx64
		     "\n",
		     vaddr, size);
	}
	return committed;
}

void WriteBacking(uint64_t vaddr, const void* data, uint64_t size) noexcept {
	if (!TryWriteBacking(vaddr, data, size)) {
		EXIT("Memory: required direct-backing write failed, addr=0x%016" PRIx64
		     " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
}

void InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (auto* renderer = Renderer(); renderer != nullptr && size != 0) {
		(void)renderer->InvalidateMemory(vaddr, size);
	}
}

bool ReadGpuTable(uint64_t vaddr, void* data, uint64_t size) {
	if (size == 0) {
		return true;
	}
	auto* renderer = Renderer();
	if (renderer == nullptr || !Graphics::GuestRange {vaddr, size}.Valid() ||
	    !Graphics::GuestGpu::IsGpuThread()) {
		return false;
	}
	return renderer->GetBufferCache().ReadTable(vaddr, size, data);
}

uint64_t GuestBackingBase() {
	return 0;
}

uint64_t GuestBackingSize() {
	return 0;
}

bool GuestBackingOffset(uint64_t /*vaddr*/, uint64_t* /*offset*/) {
	return false;
}

} // namespace Libs::LibKernel::Memory

namespace Libs::LibKernel {

uint64_t KernelGetTscFrequency() {
	LARGE_INTEGER frequency {};
	QueryPerformanceFrequency(&frequency);
	return static_cast<uint64_t>(frequency.QuadPart);
}

uint64_t KernelReadTsc() {
	LARGE_INTEGER counter {};
	QueryPerformanceCounter(&counter);
	return static_cast<uint64_t>(counter.QuadPart);
}

} // namespace Libs::LibKernel

namespace Libs::LibKernel::EventQueue {

int KernelAddEvent(KernelEqueue /*eq*/, const KernelEqueueEvent& /*event*/) {
	return KERNEL_ERROR_EBADF;
}

int KernelTriggerEvent(KernelEqueue /*eq*/, uintptr_t /*ident*/, int16_t /*filter*/, void* /*trigger_data*/) {
	return KERNEL_ERROR_EBADF;
}

int KernelDeleteEvent(KernelEqueue /*eq*/, uintptr_t /*ident*/, int16_t /*filter*/) {
	return KERNEL_ERROR_ENOENT;
}

} // namespace Libs::LibKernel::EventQueue

namespace Libs::Loader {

bool SystemContentParamSfoGetString(const char* name, std::string* value) {
	if (std::strcmp(name, "TITLE_ID") != 0) {
		return false;
	}
	if (const char* title = std::getenv("APS5_TITLE_ID"); title != nullptr && *title != '\0') {
		*value = title;
		return true;
	}
	const auto title = GetAppTitleId_nid_postfix();
	*value           = std::string(title.value, strnlen(title.value, sizeof(title.value)));
	return !value->empty();
}

} // namespace Libs::Loader
