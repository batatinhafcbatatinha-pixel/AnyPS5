#ifndef EMULATOR_SRC_COMMON_FASTTLS_H_
#define EMULATOR_SRC_COMMON_FASTTLS_H_

// AnyPS5: a per-thread value in a Windows TLS slot. MinGW's thread_local goes through emulated TLS
// (__emutls_get_address -> pthread_getspecific) on every access; on the GPU thread's per-draw
// paths that cost several percent. Holds pointer-sized trivially copyable values that start as
// zero in every thread (nullptr, false, 0).

#include <cstdint>
#include <cstring>
#include <type_traits>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <intrin.h>
#else
#include <pthread.h>
#endif

namespace Common {

#ifndef _WIN32
// Linux: a pthread key per value (nullptr, false or 0 in every new thread). Values are kept in the
// key's pointer slot; no destructor runs at thread exit, as for the Windows slots.
template <typename T>
class FastTls final {
	static_assert(sizeof(T) <= sizeof(void*) && std::is_trivially_copyable_v<T>);

public:
	FastTls() noexcept { pthread_key_create(&m_key, nullptr); }
	~FastTls() { pthread_key_delete(m_key); }
	FastTls(const FastTls&)            = delete;
	FastTls& operator=(const FastTls&) = delete;

	[[nodiscard]] T Get() const noexcept {
		const auto raw = reinterpret_cast<uintptr_t>(pthread_getspecific(m_key));
		T          value {};
		std::memcpy(&value, &raw, sizeof(T));
		return value;
	}
	void Set(T value) noexcept {
		uintptr_t raw = 0;
		std::memcpy(&raw, &value, sizeof(T));
		pthread_setspecific(m_key, reinterpret_cast<void*>(raw));
	}

	operator T() const noexcept { return Get(); } // NOLINT(google-explicit-constructor)
	FastTls& operator=(T value) noexcept {
		Set(value);
		return *this;
	}
	T operator->() const noexcept
	    requires std::is_pointer_v<T>
	{
		return Get();
	}

private:
	pthread_key_t m_key {};
};
#else
template <typename T>
class FastTls final {
	static_assert(sizeof(T) <= sizeof(void*) && std::is_trivially_copyable_v<T>);

public:
	FastTls() noexcept: m_slot(TlsAlloc()) {}
	FastTls(const FastTls&)            = delete;
	FastTls& operator=(const FastTls&) = delete;

	// The first 64 slots live in the thread environment block (TEB TlsSlots at gs:0x1480): read and
	// written in place; later ones through the API.
	[[nodiscard]] T Get() const noexcept {
		const auto raw   = m_slot < 64 ? static_cast<uintptr_t>(__readgsqword(0x1480 + m_slot * 8))
		                               : reinterpret_cast<uintptr_t>(TlsGetValue(m_slot));
		T          value {};
		std::memcpy(&value, &raw, sizeof(T));
		return value;
	}
	void Set(T value) noexcept {
		uintptr_t raw = 0;
		std::memcpy(&raw, &value, sizeof(T));
		if (m_slot < 64) {
			__writegsqword(0x1480 + m_slot * 8, raw);
		} else {
			TlsSetValue(m_slot, reinterpret_cast<void*>(raw));
		}
	}

	operator T() const noexcept { return Get(); } // NOLINT(google-explicit-constructor)
	FastTls& operator=(T value) noexcept {
		Set(value);
		return *this;
	}
	T operator->() const noexcept
	    requires std::is_pointer_v<T>
	{
		return Get();
	}

private:
	DWORD m_slot;
};
#endif

} // namespace Common

#endif // EMULATOR_SRC_COMMON_FASTTLS_H_
