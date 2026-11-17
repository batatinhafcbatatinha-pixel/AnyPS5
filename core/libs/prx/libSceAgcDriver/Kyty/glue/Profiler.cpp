#include "common/profiler.h"

#include <windows.h>

#include <string>

// Without Tracy the profiler only names the backend's threads (visible in debuggers and crash
// reports).

namespace Profiler {

void SetThreadName(const char* name) {
	if (name == nullptr) {
		return;
	}
	const std::string text(name);
	const std::wstring wide(text.begin(), text.end());
	SetThreadDescription(GetCurrentThread(), wide.c_str());
}

void Initialize() {}

void Shutdown() {}

} // namespace Profiler
