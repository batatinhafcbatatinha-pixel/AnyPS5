// Held game-patch writes (recomp-patches "offset" writes with "hold": true): values the game
// computes after start, in its main module's data. ps5port passes them as
//   APS5_HOLD_WRITES=<rva hex>:<off hex bytes>:<on hex bytes>,...
// (rva from the start of app.exe's image). Whenever "off" is found there it is replaced by "on".
//
// Two ways:
// - Write watchpoints (Windows, opt-in: APS5_HOLD_WATCH=all or a list of writes): the 8-byte slots holding the values get a hardware write
//   watchpoint (debug registers 1-3, up to 3 slots; 0 is the signal return break of Exception.cpp) in every thread of the process; the vectored
//   handler replaces the value right after the instruction that wrote it, before the game reads it
//   back. A value that the game writes and uses at once (sizes of render targets made just after)
//   needs this: a polling thread replaces it too late. APS5_HOLD_WATCH=2,3 watches only those
//   writes (0-based, in patch order). Off by default: in title A (render size held
//   with watchpoints) the menu stayed black after the profile loaded, for a reason not found.
// - A polling thread: every 3 ms (closely until each value was replaced once, for two minutes at
//   most, when there are no watchpoints). It also arms threads created later.
// APS5_HOLD_TRACE=1 logs each change of a held address (first 64) and each replacement.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace {

struct HeldWrite {
    std::uintptr_t address = 0;
    std::vector<unsigned char> off;
    std::vector<unsigned char> on;
};

bool ParseHex(const std::string& text, std::vector<unsigned char>& bytes) {
    if (text.empty() || text.size() % 2 != 0) return false;
    bytes.clear();
    for (std::size_t index = 0; index < text.size(); index += 2) bytes.push_back(static_cast<unsigned char>(std::strtoul(text.substr(index, 2).c_str(), nullptr, 16)));
    return true;
}

std::vector<HeldWrite> Parse(const char* text, std::uintptr_t base) {
    std::vector<HeldWrite> writes;
    std::string all(text);
    std::size_t start = 0;
    while (start < all.size()) {
        auto end = all.find(',', start);
        if (end == std::string::npos) end = all.size();
        const auto entry = all.substr(start, end - start);
        start = end + 1;
        const auto first = entry.find(':');
        const auto second = entry.find(':', first + 1);
        if (first == std::string::npos || second == std::string::npos) continue;
        HeldWrite write;
        write.address = base + static_cast<std::uintptr_t>(std::strtoull(entry.substr(0, first).c_str(), nullptr, 16));
        if (!ParseHex(entry.substr(first + 1, second - first - 1), write.off) || !ParseHex(entry.substr(second + 1), write.on) || write.off.size() != write.on.size()) continue;
        writes.push_back(std::move(write));
    }
    return writes;
}

std::uint32_t Word(std::uintptr_t address) {
    std::uint32_t value = 0;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
    return value;
}

// The held writes, read by the exception handler (fixed before the handler is installed).
std::vector<HeldWrite> g_writes;
// Replacements made by the handler, per write (logged by the polling thread).
std::atomic<std::uint32_t> g_watch_hits[64];

// Watched slot of each write (bit per debug register), 0 when polled only.
std::vector<std::uint64_t> g_write_slots;

// Replaces the held values that read "off" (all, or those in the watched slots of "slots");
// returns a bit per replaced write (first 64).
std::uint64_t ReplaceAll(std::uint64_t slots = ~std::uint64_t {0}) {
    std::uint64_t replaced = 0;
    for (std::size_t index = 0; index < g_writes.size(); ++index) {
        if (slots != ~std::uint64_t {0} && (index >= g_write_slots.size() || (g_write_slots[index] & slots) == 0)) continue;
        const auto& write = g_writes[index];
        auto* target = reinterpret_cast<void*>(write.address);
        if (std::memcmp(target, write.off.data(), write.off.size()) == 0) {
            std::memcpy(target, write.on.data(), write.on.size());
            if (index < 64) replaced |= std::uint64_t {1} << index;
        }
    }
    return replaced;
}

#ifdef _WIN32

// 8-byte aligned slots watched by debug registers 1-3 (at most 3).
constexpr std::size_t FirstRegister = 1;
std::vector<std::uintptr_t> g_slots;

LONG CALLBACK WatchHandler(EXCEPTION_POINTERS* exception) {
    if (exception->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP) return EXCEPTION_CONTINUE_SEARCH;
    auto* context = exception->ContextRecord;
    const auto ours = ((std::uint64_t {1} << g_slots.size()) - 1) << FirstRegister;
    if ((context->Dr6 & ours) == 0) return EXCEPTION_CONTINUE_SEARCH;
    static const bool dry = std::getenv("APS5_HOLD_WATCH_DRY") != nullptr;  // diagnostic: count only
    const auto replaced = dry ? std::uint64_t {0} : ReplaceAll(context->Dr6 & ours);
    if (dry) g_watch_hits[63].fetch_add(1, std::memory_order_relaxed);
    for (std::size_t index = 0; index < 64; ++index) {
        if (replaced & (std::uint64_t {1} << index)) g_watch_hits[index].fetch_add(1, std::memory_order_relaxed);
    }
    context->Dr6 &= ~ours;
    return EXCEPTION_CONTINUE_EXECUTION;
}

// The slots of the selected writes (all, or the indices listed in "selection", e.g. "2,3");
// fills g_write_slots. Empty when more than 3 slots are needed.
std::vector<std::uintptr_t> Slots(const std::vector<HeldWrite>& writes, const char* selection) {
    std::set<std::size_t> chosen;
    if (selection != nullptr && std::strcmp(selection, "all") != 0) {
        for (const char* at = selection; *at != '\0';) {
            char* end = nullptr;
            chosen.insert(static_cast<std::size_t>(std::strtoul(at, &end, 10)));
            if (end == at) break;
            at = *end == ',' ? end + 1 : end;
        }
    }
    std::set<std::uintptr_t> slots;
    for (std::size_t index = 0; index < writes.size(); ++index) {
        if (!chosen.empty() && chosen.count(index) == 0) continue;
        const auto& write = writes[index];
        for (auto slot = write.address & ~std::uintptr_t {7}; slot < write.address + write.off.size(); slot += 8) slots.insert(slot);
    }
    if (slots.empty() || slots.size() > 4 - FirstRegister) return {};
    std::vector<std::uintptr_t> list(slots.begin(), slots.end());
    g_write_slots.assign(writes.size(), 0);
    for (std::size_t index = 0; index < writes.size(); ++index) {
        if (!chosen.empty() && chosen.count(index) == 0) continue;
        const auto& write = writes[index];
        for (std::size_t slot = 0; slot < list.size(); ++slot) {
            if (write.address < list[slot] + 8 && list[slot] < write.address + write.off.size()) g_write_slots[index] |= std::uint64_t {1} << (slot + FirstRegister);
        }
    }
    return list;
}

// Sets the watchpoints in one thread (write-only, 8 bytes each).
bool Arm(DWORD tid) {
    HANDLE thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, FALSE, tid);
    if (thread == nullptr) return false;
    if (SuspendThread(thread) == static_cast<DWORD>(-1)) {
        CloseHandle(thread);
        return false;
    }
    CONTEXT context {};
    context.ContextFlags = CONTEXT_DEBUG_REGISTERS;
    bool armed = false;
    if (GetThreadContext(thread, &context)) {
        DWORD64* registers[] = {&context.Dr0, &context.Dr1, &context.Dr2, &context.Dr3};
        for (std::size_t slot = 0; slot < g_slots.size(); ++slot) {
            const auto index = slot + FirstRegister;
            *registers[index] = g_slots[slot];
            context.Dr7 &= ~((DWORD64 {3} << (index * 2)) | (DWORD64 {0xF} << (16 + index * 4)));
            // local enable; R/W = 01 (data writes); LEN = 10 (8 bytes)
            context.Dr7 |= (DWORD64 {1} << (index * 2)) | (DWORD64 {0x9} << (16 + index * 4));
        }
        context.ContextFlags = CONTEXT_DEBUG_REGISTERS;
        armed = SetThreadContext(thread, &context) != 0;
    }
    ResumeThread(thread);
    CloseHandle(thread);
    return armed;
}

// Arms the threads of the process not armed yet; returns how many were armed now.
int ArmNewThreads(std::set<DWORD>& armed) {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return 0;
    const auto pid = GetCurrentProcessId();
    THREADENTRY32 entry {};
    entry.dwSize = sizeof(entry);
    std::set<DWORD> alive;
    int count = 0;
    for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID != pid) continue;
        alive.insert(entry.th32ThreadID);
        // not this thread: it writes the held values itself
        if (entry.th32ThreadID == GetCurrentThreadId()) continue;
        if (armed.count(entry.th32ThreadID) == 0 && Arm(entry.th32ThreadID)) {
            armed.insert(entry.th32ThreadID);
            ++count;
        }
    }
    CloseHandle(snapshot);
    // forget ended threads (their ids can be reused)
    for (auto it = armed.begin(); it != armed.end();) it = alive.count(*it) ? std::next(it) : armed.erase(it);
    return count;
}

#endif

void Hold() {
    const auto& writes = g_writes;
    std::vector<bool> replaced(writes.size(), false);
    const auto started = std::chrono::steady_clock::now();
    const bool trace = std::getenv("APS5_HOLD_TRACE") != nullptr;
    std::vector<std::uint32_t> seen(writes.size());
    for (std::size_t index = 0; index < writes.size(); ++index) seen[index] = Word(writes[index].address);
    int changes = 0;
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); };
    auto note = [&](std::size_t index, const char* how) {
        if (trace) std::fprintf(stderr, "[patch] t=%.6f write %zu replaced at 0x%llx (%s)\n", elapsed(), index, static_cast<unsigned long long>(writes[index].address), how);
        if (!replaced[index]) {
            replaced[index] = true;
            std::fprintf(stderr, "[patch] held value at 0x%llx replaced\n", static_cast<unsigned long long>(writes[index].address));
        }
    };
    bool watching = false;
#ifdef _WIN32
    std::set<DWORD> armed;
    const char* watch = std::getenv("APS5_HOLD_WATCH");
    if (watch != nullptr && *watch != '\0' && std::strcmp(watch, "0") != 0) {
        g_slots = Slots(writes, watch);
        if (g_slots.empty()) {
            std::fprintf(stderr, "[patch] no watch slots or more than 3: held values are polled only\n");
        } else {
            AddVectoredExceptionHandler(1, WatchHandler);
            const int count = ArmNewThreads(armed);
            watching = count > 0;
            std::fprintf(stderr, "[patch] %zu write watchpoints in %d threads\n", g_slots.size(), count);
        }
    }
    auto next_arm = std::chrono::steady_clock::now();
#endif
    std::uint32_t hits[64] {};
    bool close = true;
    for (;;) {
        if (trace && changes < 64) {
            for (std::size_t index = 0; index < writes.size(); ++index) {
                const auto now = Word(writes[index].address);
                if (now != seen[index]) {
                    ++changes;
                    std::fprintf(stderr, "[patch] t=%.6f 0x%llx changed %u -> %u\n", elapsed(), static_cast<unsigned long long>(writes[index].address), seen[index], now);
                    seen[index] = now;
                }
            }
        }
        for (std::size_t index = 0; index < std::min<std::size_t>(writes.size(), 64); ++index) {
            const auto count = g_watch_hits[index].load(std::memory_order_relaxed);
            if (count != hits[index]) {
                hits[index] = count;
                note(index, "watchpoint");
            }
        }
        if (trace && changes < 64) {
            static std::uint32_t dry_seen = 0;
            const auto dry_hits = g_watch_hits[63].load(std::memory_order_relaxed);
            if (dry_hits != dry_seen) {
                ++changes;
                std::fprintf(stderr, "[patch] t=%.6f watch traps (dry) %u\n", elapsed(), dry_hits);
                dry_seen = dry_hits;
            }
        }
        const auto polled = ReplaceAll();
        for (std::size_t index = 0; index < std::min<std::size_t>(writes.size(), 64); ++index) {
            if (polled & (std::uint64_t {1} << index)) note(index, "poll");
        }
        if (trace) {
            for (std::size_t index = 0; index < writes.size(); ++index) seen[index] = Word(writes[index].address);
        }
#ifdef _WIN32
        if (!g_slots.empty() && std::chrono::steady_clock::now() >= next_arm) {
            ArmNewThreads(armed);
            next_arm = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
        }
#endif
        if (close && !watching) {
            bool all = true;
            for (const bool done : replaced) all = all && done;
            close = !all && std::chrono::steady_clock::now() - started < std::chrono::minutes(2);
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
    }
}

struct Starter {
    Starter() {
#ifdef _WIN32
        const char* text = std::getenv("APS5_HOLD_WRITES");
        if (text == nullptr || *text == '\0') return;
        const auto base = reinterpret_cast<std::uintptr_t>(GetModuleHandleW(nullptr));
        g_writes = Parse(text, base);
        if (g_writes.empty()) return;
        std::fprintf(stderr, "[patch] holding %zu game values\n", g_writes.size());
        if (std::getenv("APS5_HOLD_TRACE") != nullptr) {
            for (const auto& write : g_writes) {
                std::fprintf(stderr, "[patch] start 0x%llx = %u\n", static_cast<unsigned long long>(write.address), Word(write.address));
            }
        }
        std::thread(Hold).detach();
#endif
    }
};

const Starter starter;

}
