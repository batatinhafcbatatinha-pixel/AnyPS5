#ifndef EMULATOR_SRC_COMMON_FRAMECLOCK_H_
#define EMULATOR_SRC_COMMON_FRAMECLOCK_H_

#include <atomic>
#include <cstdint>

// Guest flips requested so far, for code that measures its effect on the frame rate.
namespace Common::FrameClock {

inline std::atomic<uint64_t> g_flips {0};

inline void NoteFlip() noexcept {
	g_flips.fetch_add(1, std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t Flips() noexcept {
	return g_flips.load(std::memory_order_relaxed);
}

} // namespace Common::FrameClock

#endif // EMULATOR_SRC_COMMON_FRAMECLOCK_H_
