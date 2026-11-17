#include "common/assert.h"

#include "kytyGitVersion.h"

#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <stdexcept>

// AnyPS5: a failed check of the GPU backend is reported on stderr before the process ends.

namespace Common {

[[noreturn]] static void DbgReport(const char* title, std::string_view text, const char* file,
                                   int line) {
	const auto message = fmt::format("{}\n{}\n{} in {}:{}\n", KYTY_BUILD_LABEL, title, text, file, line);
	Log::WriteToConsoleAndLog(message);
	std::fflush(nullptr);
	std::_Exit(321);
}

void DbgExitIfHandler(const char* expr, const char* file, int line) {
	DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr), file, line);
}

void DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

void DbgExitHandler(const char* file, int line, std::string_view text) {
	DbgReport("--- Error ---", text, file, line);
}

} // namespace Common
