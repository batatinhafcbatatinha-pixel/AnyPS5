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
#include "common/srwLock.h"
#include "common/fastTls.h"
#include "glue/CoreInternal.hpp"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "../shader/recompiler/ThreadOwned.hpp"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "loader/systemContent.h"

#include "prx/libc/include/GuestArena.hpp"
#include "prx/libkernel/AppMetadata/include/AppMetadata.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <ucontext.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
// Linux: a page protection is the mprotect flag set (PROT_NONE, PROT_READ, PROT_READ | PROT_WRITE, ...).
using DWORD = std::uint32_t;
#endif

namespace AgcDriver::VramBackend {

namespace {

constexpr uint64_t PageBytes = 0x1000;

// One run of address space with equal state, as VirtualQuery (Windows) or /proc/self/maps (Linux) tell.
struct Region {
	uint64_t base      = 0;
	uint64_t size      = 0;
	bool     committed = false; // Linux: mapped with some access (PROT_NONE counts as not committed)
	bool     readable  = false;
	bool     writable  = false;
	DWORD    protect   = 0;
};

#ifdef _WIN32
bool QueryRegion(uint64_t address, Region& out) {
	MEMORY_BASIC_INFORMATION memory {};
	if (VirtualQuery(reinterpret_cast<void*>(address), &memory, sizeof(memory)) == 0) {
		return false;
	}
	out.base      = reinterpret_cast<uint64_t>(memory.BaseAddress);
	out.size      = memory.RegionSize;
	out.committed = memory.State == MEM_COMMIT;
	out.protect   = memory.Protect;
	out.readable  = out.committed && (memory.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
	out.writable  = (memory.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY |
	                                   PAGE_EXECUTE_WRITECOPY)) != 0;
	return true;
}
#else
// Parses one /proc/self/maps line; true when it decides the answer for `address`.
bool ParseMapsLine(const char* line, const char* end, uint64_t address, Region& out) {
	const char* cursor = line;
	const auto  hex    = [&](uint64_t& value) {
        value             = 0;
        const char* first = cursor;
        while (cursor < end) {
            const char c     = *cursor;
            const int  digit = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
            if (digit < 0) {
                break;
            }
            value = value * 16 + static_cast<uint64_t>(digit);
            ++cursor;
        }
        return cursor != first;
	};
	uint64_t start = 0, stop = 0;
	if (!hex(start) || cursor == end || *cursor++ != '-' || !hex(stop) || cursor == end || *cursor++ != ' ' ||
	    end - cursor < 4) {
		return false; // not a mapping line: keep reading
	}
	const bool r = cursor[0] == 'r', w = cursor[1] == 'w', x = cursor[2] == 'x';
	if (address >= stop) {
		return false;
	}
	out = {};
	if (address < start) { // a gap before this mapping
		out.base = address;
		out.size = start - address;
		return true;
	}
	out.base      = start;
	out.size      = stop - start;
	out.committed = r || w || x;
	out.readable  = r;
	out.writable  = w;
	out.protect   = (r ? 1u : 0u) | (w ? 2u : 0u) | (x ? 4u : 0u);
	return true;
}

// Reads /proc/self/maps with plain read(2) (no allocation: the tracker may ask from a fault handler).
bool QueryRegion(uint64_t address, Region& out) {
	const int file = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
	if (file < 0) {
		return false;
	}
	constexpr uint64_t UserEnd = uint64_t {1} << 47;
	char               buffer[4096];
	size_t             kept = 0;
	bool               done = false;
	while (!done) {
		const auto got = ::read(file, buffer + kept, sizeof(buffer) - kept);
		if (got <= 0) {
			break;
		}
		const char* end  = buffer + kept + got;
		const char* line = buffer;
		for (const char* scan = buffer; scan < end && !done; ++scan) {
			if (*scan != '\n') {
				continue;
			}
			done = ParseMapsLine(line, scan, address, out);
			line = scan + 1;
		}
		kept = static_cast<size_t>(end - line);
		if (kept == sizeof(buffer)) {
			kept = 0; // one absurdly long line: drop it
		}
		std::memmove(buffer, line, kept);
	}
	::close(file);
	if (!done) { // past the last mapping
		out      = {};
		out.base = address;
		out.size = address < UserEnd ? UserEnd - address : PageBytes;
	}
	return true;
}
#endif

// Base address of the loaded module (executable or library) that holds `inside`, or of the executable
// when null (the memory statistics print addresses relative to them).
uint64_t ModuleBase(const void* inside) {
#ifdef _WIN32
	if (inside == nullptr) {
		return reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	}
	HMODULE module = nullptr;
	GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                   reinterpret_cast<LPCWSTR>(inside), &module);
	return reinterpret_cast<uint64_t>(module);
#else
	if (inside == nullptr) {
		uint64_t base = 0;
		dl_iterate_phdr(
		    [](dl_phdr_info* info, size_t, void* data) -> int {
			    uint64_t low = UINT64_MAX;
			    for (int index = 0; index < info->dlpi_phnum; ++index) {
				    if (info->dlpi_phdr[index].p_type == PT_LOAD) {
					    low = std::min<uint64_t>(low, info->dlpi_addr + info->dlpi_phdr[index].p_vaddr);
				    }
			    }
			    *static_cast<uint64_t*>(data) = low == UINT64_MAX ? 0 : low;
			    return 1;
		    },
		    &base);
		return base;
	}
	Dl_info info {};
	return dladdr(inside, &info) != 0 ? reinterpret_cast<uint64_t>(info.dli_fbase) : 0;
#endif
}

Common::SrwMutex& ProtectionMutex() {
	static Common::SrwMutex mutex;
	return mutex;
}

// Page -> the protection the guest gave it, for every page the tracker reduced.
std::unordered_map<uint64_t, DWORD>& Originals() {
	static std::unordered_map<uint64_t, DWORD> originals;
	return originals;
}

// Guest protection of committed pages the tracker gave back (restore): the next reduction of
// those pages needs no VirtualQuery. Guest mappings and protection changes forget them.
std::unordered_map<uint64_t, DWORD>& Known() {
	static std::unordered_map<uint64_t, DWORD> known;
	return known;
}

#ifndef _WIN32
DWORD Reduce(DWORD original, Common::VirtualMemory::Mode mode) {
	if (mode == Common::VirtualMemory::Mode::NoAccess) {
		return PROT_NONE;
	}
	return original & ~static_cast<DWORD>(PROT_WRITE); // read-only (write watched), execute kept
}
#else
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
#endif

std::atomic<uint64_t> g_protect_calls {0}, g_protect_pages {0};

bool Protect(uint64_t address, uint64_t bytes, DWORD protection) {
	if (Common::DebugCounters::g_aps5_stats) {
		g_protect_calls.fetch_add(1, std::memory_order_relaxed);
		g_protect_pages.fetch_add(bytes / PageBytes, std::memory_order_relaxed);
	}
#ifdef _WIN32
	DWORD previous = 0;
	return VirtualProtect(reinterpret_cast<void*>(address), bytes, protection, &previous) != 0;
#else
	return ::mprotect(reinterpret_cast<void*>(address), bytes, static_cast<int>(protection)) == 0;
#endif
}

// A run VirtualProtect refused as a whole spans mappings (separate views): protect it one memory
// region (one mapping, equal state) at a time, not one page at a time.
std::atomic<uint64_t> g_split_runs {0};
void ProtectByRegion(uint64_t begin, uint64_t end, DWORD protection) {
	if (Common::DebugCounters::g_aps5_stats) g_split_runs.fetch_add(1, std::memory_order_relaxed);
	for (uint64_t address = begin; address < end;) {
		Region memory;
		if (!QueryRegion(address, memory)) {
			(void)Protect(address, PageBytes, protection);
			address += PageBytes;
			continue;
		}
		const auto stop = std::min(end, memory.base + memory.size);
		if (!Protect(address, stop - address, protection)) {
			for (uint64_t page = address; page < stop; page += PageBytes) (void)Protect(page, PageBytes, protection);
		}
		address = stop;
	}
}

// Applies `mode` to [begin, end), page by page in runs of equal protection. Mutex held.
std::atomic<uint64_t> g_apply_calls[3] {}, g_apply_pages[3] {}; // no access, reduced, restored
// VirtualProtect calls made by the tracker (ApplyLocked) and by opened copies (CopyOpened).
std::atomic<uint64_t> g_apply_protects {0}, g_opened_protects {0}, g_opened_copies {0};
struct ProtectDelta {
	std::atomic<uint64_t>& sum;
	uint64_t               before = g_protect_calls.load(std::memory_order_relaxed);
	~ProtectDelta() {
		if (Common::DebugCounters::g_aps5_stats) {
			sum.fetch_add(g_protect_calls.load(std::memory_order_relaxed) - before, std::memory_order_relaxed);
		}
	}
};

void ApplyLocked(uint64_t begin, uint64_t end, Common::VirtualMemory::Mode mode) {
	ProtectDelta delta {g_apply_protects};
	if (Common::DebugCounters::g_aps5_stats) {
		const int kind = mode == Common::VirtualMemory::Mode::NoAccess    ? 0
		                 : mode == Common::VirtualMemory::Mode::ReadWrite ? 2
		                                                                  : 1;
		g_apply_calls[kind].fetch_add(1, std::memory_order_relaxed);
		g_apply_pages[kind].fetch_add((end - begin) / PageBytes, std::memory_order_relaxed);
	}
	auto& originals = Originals();
	if (mode == Common::VirtualMemory::Mode::ReadWrite) {
		// Restoring: exactly the pages this map holds get their guest protection back (released
		// pages left it in NoteReleased), so no VirtualQuery is needed. A run that spans two
		// mappings fails as a whole in VirtualProtect: it is then done page by page.
		uint64_t run_begin  = begin;
		DWORD    run_target = 0;
		bool     run_open   = false;
		const auto flush    = [&](uint64_t run_end) {
			if (run_open && run_end > run_begin &&
			    !Protect(run_begin, run_end - run_begin, run_target)) {
				ProtectByRegion(run_begin, run_end, run_target);
			}
			run_open = false;
		};
		for (uint64_t page = begin; page < end; page += PageBytes) {
			const auto found = originals.find(page);
			if (found == originals.end()) {
				flush(page);
				continue;
			}
			const DWORD target = found->second;
			originals.erase(found);
			Known()[page] = target;
			if (!run_open || target != run_target) {
				flush(page);
				run_begin  = page;
				run_target = target;
				run_open   = true;
			}
		}
		flush(end);
		return;
	}
	{
		auto& known = Known();
		bool  all   = true;
		for (uint64_t page = begin; page < end && all; page += PageBytes) {
			all = originals.contains(page) || known.contains(page);
		}
		if (all) {
			uint64_t run_begin  = begin;
			DWORD    run_target = 0;
			bool     run_open   = false;
			const auto flush    = [&](uint64_t run_end) {
				if (run_open && run_end > run_begin &&
				    !Protect(run_begin, run_end - run_begin, run_target)) {
					ProtectByRegion(run_begin, run_end, run_target);
				}
				run_open = false;
			};
			for (uint64_t page = begin; page < end; page += PageBytes) {
				DWORD original = 0;
				if (const auto found = originals.find(page); found != originals.end()) {
					original = found->second;
				} else {
					original = known.at(page);
					originals.emplace(page, original);
				}
				const DWORD target = Reduce(original, mode);
				if (!run_open || target != run_target) {
					flush(page);
					run_begin  = page;
					run_target = target;
					run_open   = true;
				}
			}
			flush(end);
			return;
		}
	}
	uint64_t address = begin;
	while (address < end) {
		Region memory;
		if (!QueryRegion(address, memory)) {
			return;
		}
		const auto region_end = std::min(end, memory.base + memory.size);
#ifdef _WIN32
		if (!memory.committed) {
			for (uint64_t page = address; page < region_end; page += PageBytes) {
				originals.erase(page);
			}
			address = region_end;
			continue;
		}
#else
		// A page the tracker took to PROT_NONE still counts as committed, so its protection is known.
		bool tracked = memory.committed;
		for (uint64_t page = address; page < region_end && !tracked; page += PageBytes) {
			tracked = originals.contains(page);
		}
		if (!tracked) {
			address = region_end;
			continue;
		}
#endif
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
#ifndef _WIN32
			if (!memory.committed && !originals.contains(page)) {
				flush(page);
				continue;
			}
#endif
			if (mode == Common::VirtualMemory::Mode::ReadWrite) {
				const auto found = originals.find(page);
				if (found == originals.end()) {
					flush(page);
					continue;
				}
				target = found->second;
				originals.erase(found);
			} else {
				const auto [found, inserted] = originals.try_emplace(page, memory.protect);
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
Common::FastTls<bool> t_guarded_read;

// A plain copy of guest bytes that fails instead of faulting (no tracker read-back, no page walk).
// APS5_KYTY_MEMSTATS=1: guarded reads (calls, bytes, failures) and slow reads, every 10 s.
// Bumped by every guest release (ClampRangeSize's committed-range cache).
std::atomic<uint64_t> g_commit_generation {1};
std::atomic<uint64_t> g_clamp_calls {0}, g_clamp_hits {0}, g_clamp_queries {0};
struct MemStats {
	std::atomic<uint64_t> calls {0}, bytes {0}, failed {0}, slow {0}, slow_bytes {0};
	std::atomic<int64_t>  next {0};
};
MemStats& Stats() {
	static MemStats stats;
	return stats;
}
// Reads per calling place (return address), for the 10 s report.
struct CallerSlot {
	std::atomic<uint64_t> at {0}, calls {0}, bytes {0};
};
constexpr size_t CallerSlots = 64;
CallerSlot* Callers() {
	static CallerSlot slots[CallerSlots];
	return slots;
}
void NoteCaller(void* at, uint64_t bytes) {
	static const bool enabled = std::getenv("APS5_KYTY_MEMSTATS") != nullptr;
	if (!enabled) {
		return;
	}
	const auto key  = reinterpret_cast<uint64_t>(at);
	auto*      slot = Callers();
	for (size_t probe = 0; probe < CallerSlots; ++probe) {
		auto& entry = slot[(key / 16 + probe) % CallerSlots];
		auto  seen  = entry.at.load(std::memory_order_relaxed);
		if (seen == 0 && entry.at.compare_exchange_strong(seen, key)) {
			seen = key;
		}
		if (seen == key) {
			entry.calls.fetch_add(1, std::memory_order_relaxed);
			entry.bytes.fetch_add(bytes, std::memory_order_relaxed);
			return;
		}
	}
}
void ReportCallers() {
	const uint64_t self = ModuleBase(reinterpret_cast<const void*>(&ReportCallers));
	struct Row {
		uint64_t at, calls, bytes;
	};
	std::vector<Row> rows;
	for (size_t index = 0; index < CallerSlots; ++index) {
		auto& entry = Callers()[index];
		const auto at = entry.at.load(std::memory_order_relaxed);
		if (at != 0) {
			rows.push_back({at, entry.calls.exchange(0), entry.bytes.exchange(0)});
		}
	}
	std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.bytes > b.bytes; });
	std::string text = "[memstats] callers (rva in libSceAgcDriver: calls, MiB):";
	for (size_t index = 0; index < rows.size() && index < 8; ++index) {
		text += fmt::format(" {:#x}: {} {}", rows[index].at - self, rows[index].calls,
		                    rows[index].bytes >> 20);
	}
	Log::WriteToConsoleAndLog(text + std::string(1, char(10)));
}
void ReportWaitSites() {
	const uint64_t self = ModuleBase(reinterpret_cast<const void*>(&ReportWaitSites));
	std::string text = "[memstats] scheduler waits (rva: count, ms):";
	for (auto& site: Common::DebugCounters::g_wait_sites) {
		const auto at = site.at.load(std::memory_order_relaxed);
		const auto count = site.count.exchange(0);
		const auto ns = site.ns.exchange(0);
		if (at != 0 && count != 0) {
			text += fmt::format(" {:#x}: {} {}", at - self, count, ns / 1'000'000);
		}
	}
	Log::WriteToConsoleAndLog(text + std::string(1, char(10)));
}
template <size_t Slots>
void ReportSites(const char* title, Common::DebugCounters::SiteTable<Slots>& table, bool rva, bool region_flag, int mode = 0) {
	const uint64_t self = ModuleBase(reinterpret_cast<const void*>(&ReportCallers));
	struct Row {
		uint64_t key, count, bytes;
	};
	std::vector<Row> rows;
	for (auto& slot: table.slots) {
		const auto key = slot.key.load(std::memory_order_relaxed);
		const auto count = slot.count.exchange(0);
		if (key != 0 && count != 0) rows.push_back({key & ~(1ull << 63), count, slot.bytes.exchange(0)});
	}
	std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.bytes != b.bytes ? a.bytes > b.bytes : a.count > b.count; });
	std::string text = title;
	for (size_t index = 0; index < rows.size() && index < 12; ++index) {
		const auto& row = rows[index];
		if (rva && mode == 0) {
			text += row.key == 1 ? fmt::format(" guest-fault: {} {}KiB", row.count, row.bytes >> 10)
			                     : fmt::format(" {:#x}: {} {}KiB", row.key - self, row.count, row.bytes >> 10);
		} else if (mode == 3) {
			text += fmt::format(" {}{}{}{}: {} {}ms", (row.key & 8) ? "gpu" : "guest", (row.key & 4) ? "+bytes" : "",
			                    (row.key & 2) ? "+eager" : "", (row.key & 1) ? "+sync" : "", row.count, row.bytes / 1000000);
		} else if (mode == 4) {
			text += fmt::format(" {:#x}{}{}: {}", (row.key >> 2) << 20, (row.key & 2) ? "g" : "c", (row.key & 1) ? "w" : "r", row.count);
		} else if (mode == 7) {
			constexpr const char* writer[] = {"done", "submitted", "open", "?"};
			text += fmt::format(" {:#x}{}{}{}/{}: {} {}ms", (row.key >> 5) << 20, (row.key & 16) ? "g" : "c",
			                    (row.key & 8) ? "+bytes" : "", (row.key & 4) ? "w" : "r", writer[row.key & 3],
			                    row.count, row.bytes / 1000000);
		} else if (mode == 6) {
			text += fmt::format(" {}: {}", row.key, row.count);
		} else if (mode == 5) {
			const auto exe = ModuleBase(nullptr);
			text += fmt::format(" app+{:#x}: {}", row.key - exe, row.count);
		} else if (region_flag) {
			text += fmt::format(" {:#x}{}: {} {}KiB", (row.key >> 1) << 20, (row.key & 1) ? "e" : "s", row.count, row.bytes >> 10);
		} else {
			text += fmt::format(" {:#x}: {} {}KiB", row.key << 20, row.count, row.bytes >> 10);
		}
	}
	Log::WriteToConsoleAndLog(text + std::string(1, char(10)));
}
void ReportStats() {
	static const bool enabled = [] {
		const bool on = std::getenv("APS5_KYTY_MEMSTATS") != nullptr;
		if (on) {
			Common::DebugCounters::SetEnabled(true);
		}
		return on;
	}();
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
	ReportCallers();
	ReportWaitSites();
	Log::WriteToConsoleAndLog(fmt::format(
	    "[memstats] VirtualProtect: {} calls, {} pages; tracker changes (calls/pages): no access {}/{}, "
	    "write-watched {}/{}, restored {}/{}; runs split by mapping {}; VirtualProtect calls by the tracker {}, "
	    "by {} opened copies {}\n",
	    g_protect_calls.exchange(0), g_protect_pages.exchange(0), g_apply_calls[0].exchange(0),
	    g_apply_pages[0].exchange(0), g_apply_calls[1].exchange(0), g_apply_pages[1].exchange(0),
	    g_apply_calls[2].exchange(0), g_apply_pages[2].exchange(0), g_split_runs.exchange(0),
	    g_apply_protects.exchange(0), g_opened_copies.exchange(0), g_opened_protects.exchange(0)));
	ReportSites("[memstats] opened copies per MiB (e write: count, size):", Common::DebugCounters::g_opened_regions, false, true);
	ReportSites("[memstats] readback sites (rva or guest-fault: count, size):", Common::DebugCounters::g_readback_sites, true, false);
	ReportSites("[memstats] downloads per MiB (s sync, e eager: count, size):", Common::DebugCounters::g_download_regions, false, true);
	ReportSites("[memstats] uploads per MiB (count, size):", Common::DebugCounters::g_upload_regions, false, false);
	ReportSites("[memstats] read-back outcomes (count, time):", Common::DebugCounters::g_readback_outcomes, false, false, 3);
	ReportSites("[memstats] early compute reasons (1 off, 2 short, 3 no writer, 4 no program, 5 known, 6 counted: count):",
	            Common::DebugCounters::g_early_reasons, false, false, 6);
	ReportSites("[memstats] faults per MiB (c guest/g gpu thread, r/w: count):", Common::DebugCounters::g_fault_regions, false, false, 4);
	ReportSites("[memstats] guest fault rips (count):", Common::DebugCounters::g_fault_rips, false, false, 5);
	ReportSites("[memstats] read-back sync downloads (MiB, c guest/g gpu thread, +bytes, r/w, writer: count, time):",
	            Common::DebugCounters::g_sync_downloads, false, false, 7);
	Log::WriteToConsoleAndLog(fmt::format("[memstats] gpu-modified checks {} regions {}; cpu-modified checks {} regions {}",
	                                      Libs::Graphics::g_gpumod_calls.exchange(0), Libs::Graphics::g_gpumod_regions.exchange(0),
	                                      Libs::Graphics::g_cpumod_calls.exchange(0), Libs::Graphics::g_cpumod_regions.exchange(0)) +
	                          std::string(1, char(10)));
	{
		static uint64_t last_generation = 0;
		const auto      generation      = g_commit_generation.load();
		Log::WriteToConsoleAndLog(fmt::format("[memstats] clamp: calls {} hits {} VirtualQuery {} invalidations {}",
		                                      g_clamp_calls.exchange(0), g_clamp_hits.exchange(0),
		                                      g_clamp_queries.exchange(0), generation - last_generation) +
		                          std::string(1, char(10)));
		last_generation = generation;
	}
	using Common::DebugCounters::Counter;
	static uint64_t previous[static_cast<int>(Counter::Count)] {};
	auto delta = [](Counter counter) {
		const auto value = Common::DebugCounters::Get(counter);
		const auto diff  = value - previous[static_cast<int>(counter)];
		previous[static_cast<int>(counter)] = value;
		return diff;
	};
	const auto draws = delta(Counter::Draws);
	const auto dispatches = delta(Counter::Dispatches);
	const auto submits = delta(Counter::QueueSubmits);
	const auto waits = delta(Counter::GpuWaits);
	const auto wait_ms = delta(Counter::GpuWaitNs) / 1'000'000;
	const auto faults = delta(Counter::GpuFaults);
	const auto tex_up = delta(Counter::TextureUploadBytes) >> 20;
	const auto tex_down = delta(Counter::TextureDownloadBytes) >> 20;
	const auto buf_up = delta(Counter::BufferUploadBytes) >> 20;
	const auto buf_down = delta(Counter::BufferDownloadBytes) >> 20;
	const auto meta = delta(Counter::ColorMetadataReadbacks);
	const auto meta_ms = delta(Counter::ColorMetadataReadbackNs) / 1'000'000;
	const auto busy_ms = delta(Counter::GpuThreadBusyNs) / 1'000'000;
	static uint64_t last_epoch = 0, last_frame = 0;
	const auto epoch = Libs::Graphics::TrackerEpoch(), frame = Libs::Graphics::TrackerFrame();
	const auto epochs = epoch - last_epoch, flips = frame - last_frame;
	last_epoch = epoch;
	last_frame = frame;
	const auto hot_mib  = Libs::Graphics::g_hot_upload_bytes.exchange(0) >> 20;
	const auto cold_mib = Libs::Graphics::g_cold_upload_bytes.exchange(0) >> 20;
	Log::WriteToConsoleAndLog(fmt::format(
	    "[counters] 10 s: draws {} dispatches {} submits {} gpu-waits {} ({} ms) faults {} tex up {} MiB "
	    "down {} MiB buf up {} MiB down {} MiB meta-readbacks {} ({} ms) gpu-thread busy {} ms frames {} "
	    "guest submissions {} dirty pages uploaded: hot {} MiB, other {} MiB; unchanged {} MiB changed {} MiB",
	    draws, dispatches, submits, waits, wait_ms, faults, tex_up, tex_down, buf_up, buf_down, meta,
	    meta_ms, busy_ms, flips, epochs, hot_mib, cold_mib,
	    Libs::Graphics::g_upload_same_bytes.exchange(0) >> 20,
	    Libs::Graphics::g_upload_changed_bytes.exchange(0) >> 20) + std::string(1, char(10)));
}

bool GuardedRead(uint64_t address, void* data, uint64_t bytes) {
	t_guarded_read = true;
	const bool copied = GuestArena::GuestArenaCopyOwned_nid_no_patch(address, data, bytes);
	t_guarded_read = false;
	if (!Common::DebugCounters::g_aps5_stats) {
		return copied;
	}
	auto& stats = Stats();
	stats.calls.fetch_add(1, std::memory_order_relaxed);
	stats.bytes.fetch_add(bytes, std::memory_order_relaxed);
	if (!copied) {
		stats.failed.fetch_add(1, std::memory_order_relaxed);
	}
	ReportStats();
	return copied;
}

// Copies to or from the shared backing of a guest range, ignoring the guest page protections; false
// where the range has no single backing (the caller then opens the pages).
bool BackingCopy(uint64_t address, void* buffer, uint64_t bytes, bool write) {
#ifdef _WIN32
	return GuestArena::GuestArenaBackingCopy_nid_no_patch(address, buffer, bytes, write);
#else
	int      file   = -1;
	uint64_t offset = 0;
	if (!GuestArena::GuestArenaSharedBacking_nid_postfix(address, bytes, &file, &offset) || file < 0) {
		return false;
	}
	auto*    data = static_cast<uint8_t*>(buffer);
	uint64_t done = 0;
	while (done < bytes) {
		const auto got = write ? ::pwrite(file, data + done, bytes - done, static_cast<off_t>(offset + done))
		                       : ::pread(file, data + done, bytes - done, static_cast<off_t>(offset + done));
		if (got <= 0) {
			return false;
		}
		done += static_cast<uint64_t>(got);
	}
	return true;
#endif
}

bool Tracked(uint64_t address) {
	std::lock_guard lock(ProtectionMutex());
	return Originals().contains(address & ~(PageBytes - 1));
}

bool Readable(uint64_t address, uint64_t bytes) {
	const auto end = address + bytes;
	while (address < end) {
		Region memory;
		if (!QueryRegion(address, memory) || !memory.readable) {
			return false;
		}
		address = memory.base + memory.size;
	}
	return true;
}

// Copies with the pages opened to the guest's protection for the duration (private memory has
// no shared backing to go through). A guest store racing on these pages escapes the tracker.
bool CopyOpened(uint64_t address, void* data, uint64_t bytes, bool write) {
	std::lock_guard lock(ProtectionMutex());
	ProtectDelta    delta {g_opened_protects};
	if (Common::DebugCounters::g_aps5_stats) {
		g_opened_copies.fetch_add(1, std::memory_order_relaxed);
		Common::DebugCounters::g_opened_regions.Note((address >> 20) << 1 | (write ? 1u : 0u), bytes);
	}
	auto&           originals = Originals();
	const auto      begin     = address & ~(PageBytes - 1);
	const auto      end       = (address + bytes + PageBytes - 1) & ~(PageBytes - 1);
	// Runs of pages opened together, with the protection they had: one VirtualQuery per region of
	// equal protection and one VirtualProtect per run, not three calls per page (a 512 KiB
	// download window was 384 system calls under this mutex).
	struct Run {
		uint64_t first, bytes;
		DWORD    protection;
	};
	std::vector<Run> reduced;
	const auto       restore = [&] {
		for (const auto& run: reduced) {
			if (!Protect(run.first, run.bytes, run.protection)) {
				for (uint64_t page = run.first; page < run.first + run.bytes; page += PageBytes) {
					(void)Protect(page, PageBytes, run.protection);
				}
			}
		}
	};
	// The protection a reduced page opens to, or 0 for a page the tracker left alone.
	const auto target = [&](uint64_t page) -> DWORD {
		const auto found = originals.find(page);
		return found == originals.end() ? 0 : write ? PAGE_READWRITE : found->second;
	};
	for (uint64_t page = begin; page < end;) {
		Region memory;
#ifdef _WIN32
		const bool queried = QueryRegion(page, memory) && memory.committed;
#else
		// A page the tracker took to PROT_NONE is still the guest's: only unmapped space refuses.
		const bool queried = QueryRegion(page, memory) && (memory.committed || originals.contains(page));
#endif
		if (!queried) {
			restore();
			return false;
		}
		const auto region_end = std::min(end, memory.base + memory.size);
		while (page < region_end) {
			const DWORD opened = target(page);
			uint64_t    last   = page + PageBytes;
			while (last < region_end && target(last) == opened) last += PageBytes;
			if (opened != 0) {
				if (!Protect(page, last - page, opened)) {
					restore();
					return false;
				}
				reduced.push_back({page, last - page, memory.protect});
			} else if (write ? !memory.writable : !memory.readable) {
				restore();
				return false;
			}
			page = last;
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

namespace {

// What the tracker needs of a fault, whatever the system reports it as: the access kind (0 read, 1 write,
// 8 execute), the address, and the registers the SRT walker's loads are skipped through.
struct FaultState {
	uint64_t  kind    = 0;
	uint64_t  address = 0;
	uint64_t* rip     = nullptr;
	uint64_t* rax     = nullptr;
	uint64_t  r10     = 0;
};

bool HandleFaultState(const FaultState& state) noexcept {
	const auto kind    = state.kind;
	const auto address = state.address;
	auto*       renderer = Internal::Renderer();
	if (renderer == nullptr || t_guarded_read) {
		return false;
	}
	namespace IR       = Libs::Graphics::ShaderRecompiler::IR;
	const auto skip_load = [&state](uint32_t length, uint32_t value) {
		*state.rax = value;
		*state.rip += length;
	};
	if (kind == 0) {
		const auto length = IR::SrtWalkerLoadLength(*state.rip);
		if (length != 0 && IR::WalkerSkipsGpuWritten()) {
			skip_load(length, 0);
			IR::MarkWalkerTainted();
			return true;
		}
		uint32_t value = 0;
		if (length != 0) {
			IR::SrtWalkerNoteProtectedPage(state.r10);
		}
		if (length != 0 && Libs::Graphics::GuestGpu::IsGpuThread() &&
		    renderer->GetBufferCache().TryCurrentWalkerLoad(state.r10, value)) {
			skip_load(length, value);
			return true;
		}
	}
	if (!Tracked(address)) {
		return false;
	}
	if (Common::DebugCounters::g_aps5_stats) {
		const bool gpu = Libs::Graphics::GuestGpu::IsGpuThread();
		Common::DebugCounters::g_fault_regions.Note((address >> 20) << 2 | (gpu ? 2u : 0u) | (kind == 1 ? 1u : 0u), 0);
		if (!gpu) Common::DebugCounters::g_fault_rips.Note(*state.rip, 0);
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
	if (const auto length = IR::SrtWalkerLoadLength(*state.rip); length != 0) {
		skip_load(length, 0);
		return true;
	}
	return false;
}

} // namespace

#ifdef _WIN32
bool Core::HandleFault(void* exception_pointers) noexcept {
	auto* const info   = static_cast<EXCEPTION_POINTERS*>(exception_pointers);
	const auto* record = info->ExceptionRecord;
	auto*       context = info->ContextRecord;
	FaultState  state;
	state.kind    = record->ExceptionInformation[0];
	state.address = static_cast<uint64_t>(record->ExceptionInformation[1]);
	state.rip     = reinterpret_cast<uint64_t*>(&context->Rip);
	state.rax     = reinterpret_cast<uint64_t*>(&context->Rax);
	state.r10     = context->R10;
	return HandleFaultState(state);
}
#else
bool Core::HandleFault(void* signal_info, void* signal_context) noexcept {
	const auto* info    = static_cast<const siginfo_t*>(signal_info);
	auto*       context = static_cast<ucontext_t*>(signal_context);
	if (info == nullptr || context == nullptr) {
		return false;
	}
	auto&       regs = context->uc_mcontext.gregs;
	// x86 page-fault error code: bit 1 write, bit 4 instruction fetch.
	const auto  error = static_cast<uint64_t>(regs[REG_ERR]);
	FaultState  state;
	state.kind    = (error & 2u) != 0 ? 1 : (error & 16u) != 0 ? 8 : 0;
	state.address = reinterpret_cast<uint64_t>(info->si_addr);
	state.rip     = reinterpret_cast<uint64_t*>(&regs[REG_RIP]);
	state.rax     = reinterpret_cast<uint64_t*>(&regs[REG_RAX]);
	state.r10     = static_cast<uint64_t>(regs[REG_R10]);
	return HandleFaultState(state);
}
#endif

void Core::NoteMapped(std::uint64_t address, std::size_t bytes) {
	g_commit_generation.fetch_add(1, std::memory_order_acq_rel);
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
			Known().erase(page);
		}
	}
	if (Libs::Graphics::GuestRange {address, bytes}.Valid()) {
		renderer->MapMemory(address, bytes);
		(void)renderer->InvalidateMemory(address, bytes);
	}
}

void Core::NoteReleased(std::uint64_t address, std::size_t bytes) {
	g_commit_generation.fetch_add(1, std::memory_order_acq_rel);
	auto* renderer = Internal::Renderer();
	if (renderer == nullptr || bytes == 0 || !Libs::Graphics::GuestRange {address, bytes}.Valid()) {
		return;
	}
	renderer->UnmapMemory(address, bytes);
	std::lock_guard lock(ProtectionMutex());
	auto&           originals = Originals();
	for (uint64_t page = address & ~(PageBytes - 1); page < address + bytes; page += PageBytes) {
		originals.erase(page);
		Known().erase(page);
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
#ifndef _WIN32
		// A system call writing into a page the tracker reduced fails with EFAULT instead of faulting:
		// the pages of the range go back to the guest's protection first (the write invalidated them).
		std::lock_guard lock(ProtectionMutex());
		const auto      begin = address & ~(PageBytes - 1);
		const auto      end   = (address + bytes + PageBytes - 1) & ~(PageBytes - 1);
		bool            any   = false;
		for (uint64_t page = begin; page < end && !any; page += PageBytes) {
			any = Originals().contains(page);
		}
		if (any) {
			ApplyLocked(begin, end, Common::VirtualMemory::Mode::ReadWrite);
		}
#endif
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
	if (AgcDriver::VramBackend::BackingCopy(vaddr, const_cast<void*>(data), size, true)) {
		return true;
	}
	return AgcDriver::VramBackend::CopyOpened(vaddr, const_cast<void*>(data), size, true);
}

bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
	if (size == 0) {
		return true;
	}
	AgcDriver::VramBackend::NoteCaller(__builtin_return_address(0), size);
	// Fast path (most reads): the pages are readable as mapped.
	if (AgcDriver::VramBackend::GuardedRead(vaddr, data, size)) {
		return true;
	}
	AgcDriver::VramBackend::Stats().slow.fetch_add(1, std::memory_order_relaxed);
	AgcDriver::VramBackend::Stats().slow_bytes.fetch_add(size, std::memory_order_relaxed);
	if (AgcDriver::VramBackend::BackingCopy(vaddr, data, size, false)) {
		return true;
	}
	if (AgcDriver::VramBackend::Readable(vaddr, size)) {
		std::memcpy(data, reinterpret_cast<const void*>(vaddr), size);
		return true;
	}
	return AgcDriver::VramBackend::CopyOpened(vaddr, data, size, false);
}

bool TryReadBackingStrided(uint64_t first, uint64_t stride, uint32_t count, uint8_t* out) {
	const uint64_t last_index = count - 1u;
	if (out == nullptr || count == 0 || first == UINT64_MAX ||
	    (stride != 0 && last_index > (UINT64_MAX - 1u - first) / stride)) {
		return false;
	}
	AgcDriver::VramBackend::NoteCaller(__builtin_return_address(0), count);
	// A short span is read in one copy and the bytes picked from it. The span may hold unbacked
	// bytes between the addresses, so a failed copy falls back to one read per address.
	constexpr uint64_t SpanLimit = 64 * 1024;
	const uint64_t     span      = stride * last_index + 1u;
	if (span <= SpanLimit) {
		thread_local std::vector<uint8_t>* bytesSlot = nullptr;
		auto&                              bytes     = ShaderRecompiler::ThreadOwned(bytesSlot);
		bytes.resize(span);
		if (AgcDriver::VramBackend::GuardedRead(first, bytes.data(), span)) {
			for (uint32_t index = 0; index < count; ++index) {
				out[index] = bytes[stride * index];
			}
			return true;
		}
	}
	for (uint32_t index = 0; index < count; ++index) {
		if (!TryReadBacking(first + stride * index, out + index, 1)) {
			return false;
		}
	}
	return true;
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
		AgcDriver::VramBackend::Region memory;
		if (!AgcDriver::VramBackend::QueryRegion(vaddr, memory)) {
			return false;
		}
		const auto region_end = std::min(end, memory.base + memory.size);
		const auto bytes = region_end - vaddr;
		// As KytyPS5: gaps (reserved, free or inaccessible guest pages) read as zero.
		static_cast<void>(zero_unmapped);
		// Linux: a page the tracker took to PROT_NONE is still the guest's (TryReadBacking opens it).
		const bool mapped = memory.committed || AgcDriver::VramBackend::Tracked(vaddr);
		if (!mapped || !TryReadBacking(vaddr, out, bytes)) {
			std::memset(out, 0, bytes);
		}
		out += bytes;
		vaddr = region_end;
	}
	return true;
}

bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size) {
	AgcDriver::VramBackend::NoteCaller(__builtin_return_address(0), size);
	return CopySparse(vaddr, data, size, false);
}

bool CopySparseMemory(uint64_t vaddr, void* data, uint64_t size) {
	AgcDriver::VramBackend::NoteCaller(__builtin_return_address(0), size);
	return CopySparse(vaddr, data, size, true);
}

// Ranges ClampRangeSize found committed, per thread; a guest release drops them all.
struct CommittedRange {
	uint64_t begin = 0, end = 0;
	bool     bounded = false;  // end is the first uncommitted byte
};
struct CommittedCache {
	CommittedRange ranges[16];
	uint64_t       generation = 0;
	uint32_t       next       = 0;
};
Common::FastTls<CommittedCache*> t_committed_cache;

uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size) {
	// Vertex ranges are clamped at every draw: a range known committed needs no VirtualQuery.
	if (Common::DebugCounters::g_aps5_stats) AgcDriver::VramBackend::g_clamp_calls.fetch_add(1, std::memory_order_relaxed);
	const auto generation = AgcDriver::VramBackend::g_commit_generation.load(std::memory_order_acquire);
	auto*      cache      = t_committed_cache.Get();
	if (cache == nullptr) {
		cache = new CommittedCache();
		t_committed_cache.Set(cache);
	}
	auto& t_committed = cache->ranges;
	if (cache->generation != generation) {
		for (auto& range: t_committed) {
			range = {};
		}
		cache->generation = generation;
	}
	for (const auto& range: t_committed) {
		if (range.begin <= vaddr && vaddr < range.end) {
			if (vaddr + size <= range.end) {
				if (Common::DebugCounters::g_aps5_stats) AgcDriver::VramBackend::g_clamp_hits.fetch_add(1, std::memory_order_relaxed);
				return size;
			}
			if (range.bounded) {
				if (Common::DebugCounters::g_aps5_stats) AgcDriver::VramBackend::g_clamp_hits.fetch_add(1, std::memory_order_relaxed);
				return range.end - vaddr;
			}
		}
	}
	// The renderer's map of guest mappings answers most ranges (ring buffers move every draw).
	if (auto* renderer = Renderer(); renderer != nullptr && size != 0 && renderer->IsMapped(vaddr, size)) {
		if (Common::DebugCounters::g_aps5_stats) AgcDriver::VramBackend::g_clamp_hits.fetch_add(1, std::memory_order_relaxed);
		return size;
	}
	uint64_t       committed = 0;
	const uint64_t end       = vaddr + size;
	uint64_t       address   = vaddr;
	while (address < end) {
		AgcDriver::VramBackend::Region memory;
		if (Common::DebugCounters::g_aps5_stats) AgcDriver::VramBackend::g_clamp_queries.fetch_add(1, std::memory_order_relaxed);
		if (!AgcDriver::VramBackend::QueryRegion(address, memory) ||
		    !(memory.committed || AgcDriver::VramBackend::Tracked(address))) {
			break;
		}
		const auto region_end = std::min(end, memory.base + memory.size);
		committed += region_end - address;
		address = region_end;
	}
	if (committed != 0) {
		t_committed[cache->next++ % 16] = {vaddr, vaddr + committed, committed != size};
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

#ifdef _WIN32
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
#else
// Nanoseconds of the monotonic clock.
uint64_t KernelGetTscFrequency() {
	return 1'000'000'000ull;
}

uint64_t KernelReadTsc() {
	timespec now {};
	clock_gettime(CLOCK_MONOTONIC, &now);
	return static_cast<uint64_t>(now.tv_sec) * 1'000'000'000ull + static_cast<uint64_t>(now.tv_nsec);
}
#endif

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
