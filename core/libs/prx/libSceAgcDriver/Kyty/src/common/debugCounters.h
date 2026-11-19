#ifndef KYTY_COMMON_DEBUG_COUNTERS_H_
#define KYTY_COMMON_DEBUG_COUNTERS_H_

#include <atomic>
#include <cstdlib>
#include <chrono>
#include <cstdint>

namespace Common::DebugCounters {

static_assert(std::atomic<uint64_t>::is_always_lock_free && std::atomic<bool>::is_always_lock_free);

// Collect events only while the panel is visible. Relaxed, lock-free atomics keep updates
// cheap and safe in signal handlers. Rates come from comparing two snapshots.
enum class Counter {
	// GPU work.
	Draws,
	DrawInstances,
	IndirectDraws,
	Dispatches,
	QueueSubmits,
	GpuWaits, // CPU blocked on a GPU timeline semaphore.
	GpuWaitNs,
	GpuFaults, // Guest writes to GPU-tracked memory.
	ShadersCompiled,
	ShaderCompileNs,
	PipelinesCreated,
	PipelineCreateNs,
	// Caches.
	TextureUploadBytes,
	TextureDownloadBytes,
	TextureEvictions,
	ColorMetadataReadbacks, // CPU blocked reading back GPU-written DCC/CMASK metadata.
	ColorMetadataReadbackNs,
	ColorMetadataChecks, // GPU clear-key checks of DCC/CMASK metadata.
	BufferUploadBytes,
	BufferDownloadBytes,
	BufferEvictions,
	// Presentation.
	FlipsRequested,
	FlipsRejected, // Guest flip queue was full.
	LateVblanks,   // The present thread fell behind the vblank rate.
	SwapchainRecreations,
	DroppedPresents,
	// Emulator.
	EmulatedInstructions, // Host CPU lacked an instruction the guest used.
	UnresolvedImportCalls,
	GuestFileReadBytes,
	GuestFileWriteBytes,
	AudioUnderruns,
	// Overlapping host elapsed timings, including waits.
	GpuThreadBusyNs,
	PresentNs, // Includes image acquire and overlay preparation.
	PresentAcquireNs,
	OverlayNs,
	Count
};

enum class Peak { ShaderCompileNs, PipelineCreateNs, Count };

namespace Detail {
inline std::atomic<bool>     g_enabled {false};
inline std::atomic<uint64_t> g_counters[static_cast<int>(Counter::Count)] {};
inline std::atomic<uint64_t> g_peaks[static_cast<int>(Peak::Count)] {};
} // namespace Detail

// AnyPS5: blocking waits for the GPU per calling place (return address into the caller).
struct WaitSite {
	std::atomic<uint64_t> at {0}, count {0}, ns {0};
};
inline WaitSite g_wait_sites[32];
inline void NoteWaitSite(void* at, uint64_t ns) noexcept {
	const auto key = reinterpret_cast<uint64_t>(at);
	for (size_t probe = 0; probe < 32; ++probe) {
		auto& site = g_wait_sites[(key / 16 + probe) % 32];
		auto  seen = site.at.load(std::memory_order_relaxed);
		if (seen == 0 && site.at.compare_exchange_strong(seen, key)) {
			seen = key;
		}
		if (seen == key) {
			site.count.fetch_add(1, std::memory_order_relaxed);
			site.ns.fetch_add(ns, std::memory_order_relaxed);
			return;
		}
	}
}

// AnyPS5: count and bytes per key (a return address or an address region), for the 10 s report.
struct SiteSlot {
	std::atomic<uint64_t> key {0}, count {0}, bytes {0};
};
template <size_t Slots>
struct SiteTable {
	SiteSlot slots[Slots];
	void Note(uint64_t key, uint64_t bytes) noexcept {
		key |= 1ull << 63; // 0 marks a free slot
		for (size_t probe = 0; probe < Slots; ++probe) {
			auto& slot = slots[(key / 16 + probe) % Slots];
			auto  seen = slot.key.load(std::memory_order_relaxed);
			if (seen == 0 && slot.key.compare_exchange_strong(seen, key)) {
				seen = key;
			}
			if (seen == key) {
				slot.count.fetch_add(1, std::memory_order_relaxed);
				slot.bytes.fetch_add(bytes, std::memory_order_relaxed);
				return;
			}
		}
	}
};
inline SiteTable<64>  g_readback_sites;   // ReadMemory callers: key = return address
inline SiteTable<128> g_download_regions; // key = (vaddr >> 20) << 1 | eager
inline SiteTable<128> g_upload_regions;   // key = vaddr >> 20
// key = gpu thread 8 | bytes a GPU write covers 4 | eager download in the window 2 | sync download 1; bytes = ns
inline SiteTable<16>  g_readback_outcomes;
inline SiteTable<128> g_fault_regions;    // key = (address >> 20) << 2 | gpu thread 2 | write 1
inline SiteTable<64>  g_fault_rips;       // guest-thread faults: key = rip

// AnyPS5: diagnostics of APS5_KYTY_MEMSTATS=1 (read once; off: no counter work on hot paths).
inline const bool g_aps5_stats = std::getenv("APS5_KYTY_MEMSTATS") != nullptr;

[[nodiscard]] inline bool Enabled() noexcept {
	return Detail::g_enabled.load(std::memory_order_relaxed);
}

inline void SetEnabled(bool enabled) noexcept {
	Detail::g_enabled.store(enabled, std::memory_order_relaxed);
}

inline void Add(Counter counter, uint64_t value = 1) noexcept {
	if (!Enabled()) {
		return;
	}
	Detail::g_counters[static_cast<int>(counter)].fetch_add(value, std::memory_order_relaxed);
}

class ScopedTimer final {
public:
	explicit ScopedTimer(Counter duration): ScopedTimer(Counter::Count, duration) {}
	ScopedTimer(Counter count, Counter duration, Peak peak = Peak::Count)
	    : m_count(count), m_duration(duration), m_peak(peak),
	      m_start(Enabled() ? std::chrono::steady_clock::now()
	                        : std::chrono::steady_clock::time_point {}) {}
	~ScopedTimer() {
		if (m_start == std::chrono::steady_clock::time_point {} || !Enabled()) {
			return;
		}
		const auto elapsed =
		    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                              std::chrono::steady_clock::now() - m_start)
		                              .count());
		if (m_count != Counter::Count) {
			Add(m_count);
		}
		Add(m_duration, elapsed);
		if (m_peak != Peak::Count) {
			auto& peak     = Detail::g_peaks[static_cast<int>(m_peak)];
			auto  previous = peak.load(std::memory_order_relaxed);
			while (previous < elapsed &&
			       !peak.compare_exchange_weak(previous, elapsed, std::memory_order_relaxed)) {
			}
		}
	}
	ScopedTimer(const ScopedTimer&)            = delete;
	ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
	Counter                               m_count;
	Counter                               m_duration;
	Peak                                  m_peak;
	std::chrono::steady_clock::time_point m_start;
};

[[nodiscard]] inline uint64_t Get(Counter counter) noexcept {
	return Detail::g_counters[static_cast<int>(counter)].load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t Get(Peak peak) noexcept {
	return Detail::g_peaks[static_cast<int>(peak)].load(std::memory_order_relaxed);
}

} // namespace Common::DebugCounters

#endif /* KYTY_COMMON_DEBUG_COUNTERS_H_ */
