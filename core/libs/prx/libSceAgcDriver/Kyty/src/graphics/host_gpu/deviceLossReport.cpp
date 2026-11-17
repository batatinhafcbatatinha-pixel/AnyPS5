#include "graphics/host_gpu/deviceLossReport.h"

#include "graphics/host_gpu/graphicContext.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace Libs::Graphics {

void ReportDeviceLoss(const GraphicContext& graphics, const char* where, int result) {
	static std::atomic<bool> reported {false};
	if (reported.exchange(true)) return;
	constexpr double MiB = 1024.0 * 1024.0;
	double ram_free = -1, ram_total = -1, commit = -1, commit_limit = -1;
#if defined(_WIN32)
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	if (GlobalMemoryStatusEx(&status) != 0) {
		ram_free     = static_cast<double>(status.ullAvailPhys) / MiB;
		ram_total    = static_cast<double>(status.ullTotalPhys) / MiB;
		commit_limit = static_cast<double>(status.ullTotalPageFile) / MiB;
		commit       = static_cast<double>(status.ullTotalPageFile - status.ullAvailPageFile) / MiB;
	}
#endif
	double vram_usage = -1, vram_budget = -1, host_usage = -1, host_budget = -1;
	std::array<GraphicContext::HeapUsage, VK_MAX_MEMORY_HEAPS> heaps {};
	const auto count = graphics.GetHeapUsage(heaps);
	for (uint32_t i = 0; i < count; i++) {
		auto& usage  = heaps[i].device_local ? vram_usage : host_usage;
		auto& budget = heaps[i].device_local ? vram_budget : host_budget;
		usage        = (usage < 0 ? 0 : usage) + static_cast<double>(heaps[i].usage) / MiB;
		budget       = (budget < 0 ? 0 : budget) + static_cast<double>(heaps[i].budget) / MiB;
	}
	std::printf("[device-loss] %s result=%d | RAM free %.0f / %.0f MiB | commit %.0f / %.0f MiB | "
	            "VRAM %.0f / budget %.0f MiB | host heaps %.0f / budget %.0f MiB%s\n",
	            where, result, ram_free, ram_total, commit, commit_limit, vram_usage, vram_budget,
	            host_usage, host_budget, graphics.CanReportMemoryUsage() ? "" : " (no memory budget)");
	std::fflush(stdout);
}

} // namespace Libs::Graphics
