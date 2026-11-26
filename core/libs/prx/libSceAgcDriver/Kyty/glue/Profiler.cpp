#include "common/profiler.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <string>

// Without Tracy the profiler only names the backend's threads (visible in debuggers and crash
// reports).

namespace Profiler {

void SetThreadName(const char* name) {
	if (name == nullptr) {
		return;
	}
	const std::string text(name);
#ifdef _WIN32
	const std::wstring wide(text.begin(), text.end());
	SetThreadDescription(GetCurrentThread(), wide.c_str());
#else
	// Linux limits a thread name to 15 characters.
	const std::string shortName = text.substr(0, 15);
	pthread_setname_np(pthread_self(), shortName.c_str());
#endif
}

void Initialize() {}

void Shutdown() {}

} // namespace Profiler
