#pragma once

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"

#include <atomic>
#include <cstdio>

namespace AgcDriver::Graphics {

// Device-loss hunt (Eva6): the main VkQueue needs external synchronization. Each scope that uses it
// (vkQueueSubmit / vkQueueWaitIdle) records its site; a second site entering while one is inside, or a
// site entering without the GPU mutex, is reported as [queue-race] (first 20 of each, then every 1000th).
class MainQueueUse {
public:
    explicit MainQueueUse(const char* site) noexcept : site(site) {
        const char* expected = nullptr;
        owned = Holder().compare_exchange_strong(expected, site, std::memory_order_acq_rel);
        if (!owned) {
            const auto count = Races().fetch_add(1, std::memory_order_relaxed) + 1;
            if (count <= 20 || count % 1000 == 0) std::fprintf(stderr, "[queue-race] main queue entered by %s while %s is inside (%llu so far)\n", site, expected != nullptr ? expected : "?", static_cast<unsigned long long>(count));
        }
        bool held = true;
        try {
            held = GuestMemory::GpuMutex().HeldByThisThread();
        } catch (...) {
        }
        if (!held) {
            const auto count = Unlocked().fetch_add(1, std::memory_order_relaxed) + 1;
            if (count <= 20 || count % 1000 == 0) std::fprintf(stderr, "[queue-race] main queue used by %s without the GPU mutex (%llu so far)\n", site, static_cast<unsigned long long>(count));
        }
    }
    ~MainQueueUse() {
        if (owned) Holder().store(nullptr, std::memory_order_release);
    }
    MainQueueUse(const MainQueueUse&) = delete;
    MainQueueUse& operator=(const MainQueueUse&) = delete;

private:
    static std::atomic<const char*>& Holder() noexcept {
        static std::atomic<const char*> holder{nullptr};
        return holder;
    }
    static std::atomic<unsigned long long>& Races() noexcept {
        static std::atomic<unsigned long long> races{0};
        return races;
    }
    static std::atomic<unsigned long long>& Unlocked() noexcept {
        static std::atomic<unsigned long long> unlocked{0};
        return unlocked;
    }
    const char* site;
    bool owned = false;
};

}
