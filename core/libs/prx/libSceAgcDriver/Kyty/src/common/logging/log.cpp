#include "common/logging/log.h"

#include <cstdio>
#include <cstdlib>
#include <string>

// AnyPS5: KytyPS5's log goes to stderr (no spdlog), and stays silent unless APS5_KYTY_LOG is set.

namespace Log {

void Initialize() {}

void Flush() {
	std::fflush(stderr);
}

bool IsSilent() {
	static const bool silent = std::getenv("APS5_KYTY_LOG") == nullptr;
	return silent;
}

void Write(fmt::text_style /*style*/, std::string_view text) {
	std::fwrite(text.data(), 1, text.size(), stderr);
}

void WriteToConsoleAndLog(std::string_view text) {
	std::string line("[kyty] ");
	line.append(text);
	std::fwrite(line.data(), 1, line.size(), stderr);
	std::fflush(stderr);
}

} // namespace Log
