#ifndef EMULATOR_SRC_COMMON_SRWLOCK_H_
#define EMULATOR_SRC_COMMON_SRWLOCK_H_

// AnyPS5: mutexes on Windows slim reader/writer locks. With the MinGW posix thread model
// std::mutex and std::shared_mutex go through winpthread, several times slower per lock than an
// SRW lock; the GPU thread takes some of them on every draw. Not recursive (as std::mutex);
// usable with std::lock_guard, std::unique_lock, std::scoped_lock, std::shared_lock and
// std::condition_variable_any.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace Common {

class SrwMutex final {
public:
	SrwMutex() noexcept                  = default;
	SrwMutex(const SrwMutex&)            = delete;
	SrwMutex& operator=(const SrwMutex&) = delete;

	void lock() noexcept { AcquireSRWLockExclusive(&m_lock); }
	bool try_lock() noexcept { return TryAcquireSRWLockExclusive(&m_lock) != 0; }
	void unlock() noexcept { ReleaseSRWLockExclusive(&m_lock); }

	void lock_shared() noexcept { AcquireSRWLockShared(&m_lock); }
	bool try_lock_shared() noexcept { return TryAcquireSRWLockShared(&m_lock) != 0; }
	void unlock_shared() noexcept { ReleaseSRWLockShared(&m_lock); }

private:
	SRWLOCK m_lock = SRWLOCK_INIT;
};

using SrwSharedMutex = SrwMutex;

} // namespace Common

#endif // EMULATOR_SRC_COMMON_SRWLOCK_H_
