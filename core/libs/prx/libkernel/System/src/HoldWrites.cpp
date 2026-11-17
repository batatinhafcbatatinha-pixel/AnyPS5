// Held game-patch writes (recomp-patches "offset" writes with "hold": true): values the game
// computes after start, in its main module's data. ps5port passes them as
//   APS5_HOLD_WRITES=<rva hex>:<off hex bytes>:<on hex bytes>,...
// (rva from the start of app.exe's image). A thread replaces "off" by "on" whenever it finds "off"
// there: closely until each one was replaced once (or for two minutes), then every 3 ms.
// Same behaviour as KytyPS5's game-patch hold writes (src/loader/gamePatch.cpp HoldWrites).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
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

void Hold(std::vector<HeldWrite> writes) {
    std::vector<bool> replaced(writes.size(), false);
    const auto started = std::chrono::steady_clock::now();
    bool close = true;
    for (;;) {
        for (std::size_t index = 0; index < writes.size(); ++index) {
            auto* target = reinterpret_cast<void*>(writes[index].address);
            if (std::memcmp(target, writes[index].off.data(), writes[index].off.size()) == 0) {
                std::memcpy(target, writes[index].on.data(), writes[index].on.size());
                if (!replaced[index]) {
                    replaced[index] = true;
                    std::fprintf(stderr, "[patch] held value at 0x%llx replaced\n", static_cast<unsigned long long>(writes[index].address));
                }
            }
        }
        if (close) {
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
        auto writes = Parse(text, base);
        if (writes.empty()) return;
        std::fprintf(stderr, "[patch] holding %zu game values\n", writes.size());
        std::thread(Hold, std::move(writes)).detach();
#endif
    }
};

const Starter starter;

}
