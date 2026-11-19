#include "graphics/host_gpu/renderer/gpuDiag.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics::GpuDiag {
namespace {

// Checkpoint markers are sequence numbers (never 0); the ring maps the recent ones to programs.
constexpr uint64_t RingSize = 1u << 16u;

struct Entry {
	std::atomic<uint64_t> sequence {0};
	std::atomic<uint64_t> key {0};
	std::atomic<uint64_t> tick {0};
	std::atomic<char>     kind {0};
};

std::array<Entry, RingSize>& Ring() {
	static std::array<Entry, RingSize> ring;
	return ring;
}

std::atomic<uint64_t> g_sequence {0};

std::string Describe(uint64_t marker) {
	const auto& entry = Ring()[marker % RingSize];
	char        text[96];
	if (entry.sequence.load(std::memory_order_relaxed) != marker) {
		std::snprintf(text, sizeof(text), "#%llu (overwritten)", static_cast<unsigned long long>(marker));
	} else {
		std::snprintf(text, sizeof(text), "#%llu %c:%016llx tick %llu", static_cast<unsigned long long>(marker),
		              entry.kind.load(std::memory_order_relaxed),
		              static_cast<unsigned long long>(entry.key.load(std::memory_order_relaxed)),
		              static_cast<unsigned long long>(entry.tick.load(std::memory_order_relaxed)));
	}
	return text;
}

void ReportQueue(vk::Queue queue, const char* name) {
	if (!queue || VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV == nullptr) return;
	uint32_t count = 0;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(queue, &count, nullptr);
	std::vector<VkCheckpointDataNV> data(count, VkCheckpointDataNV {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(queue, &count, data.data());
	std::string text = std::string("[gpu-diag] ") + name + " checkpoints:";
	if (count == 0) text += " none";
	uint64_t last = 0;
	for (uint32_t index = 0; index < count; ++index) {
		char stage[32];
		std::snprintf(stage, sizeof(stage), " [stage 0x%x] ", static_cast<unsigned>(data[index].stage));
		const auto marker = reinterpret_cast<uint64_t>(data[index].pCheckpointMarker);
		text += stage + Describe(marker);
		last = std::max(last, marker);
	}
	// What was recorded right after: the work the queue did not reach.
	for (uint64_t next = last + 1; last != 0 && next <= last + 6; ++next) {
		text += (next == last + 1 ? " | then " : ", ") + Describe(next);
	}
	Log::WriteToConsoleAndLog(text + "\n");
}

void ReportFault(const GraphicContext& graphics) {
	if (!graphics.device_fault_enabled || VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT == nullptr) return;
	const VkDevice device = graphics.device;
	VkDeviceFaultCountsEXT counts {VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT(device, &counts, nullptr) != VK_SUCCESS) {
		Log::WriteToConsoleAndLog("[gpu-diag] device fault: no report\n");
		return;
	}
	std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<VkDeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	VkDeviceFaultInfoEXT info {VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
	info.pAddressInfos = addresses.data();
	info.pVendorInfos  = vendors.data();
	counts.vendorBinarySize = 0; // No vendor binary dump.
	const auto result = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT(device, &counts, &info);
	std::string text = "[gpu-diag] device fault (result " + std::to_string(result) + "): " + info.description;
	for (uint32_t index = 0; index < counts.addressInfoCount; ++index) {
		char line[128];
		std::snprintf(line, sizeof(line), " | address type %d at 0x%llx (+-0x%llx)",
		              static_cast<int>(addresses[index].addressType),
		              static_cast<unsigned long long>(addresses[index].reportedAddress),
		              static_cast<unsigned long long>(addresses[index].addressPrecision));
		text += line;
	}
	for (uint32_t index = 0; index < counts.vendorInfoCount; ++index) {
		char line[320];
		std::snprintf(line, sizeof(line), " | vendor 0x%llx/0x%llx %s",
		              static_cast<unsigned long long>(vendors[index].vendorFaultCode),
		              static_cast<unsigned long long>(vendors[index].vendorFaultData), vendors[index].description);
		text += line;
	}
	Log::WriteToConsoleAndLog(text + "\n");
}

} // namespace

bool Wanted() {
	static const bool wanted = std::getenv("APS5_KYTY_GPU_DIAG") != nullptr;
	return wanted;
}

void Checkpoint(const GraphicContext& graphics, vk::CommandBuffer command, uint64_t key, char kind,
                uint64_t tick) {
	if (!graphics.diagnostic_checkpoints_enabled) return;
	const auto sequence = ++g_sequence;
	auto&      entry    = Ring()[sequence % RingSize];
	entry.key.store(key, std::memory_order_relaxed);
	entry.tick.store(tick, std::memory_order_relaxed);
	entry.kind.store(kind, std::memory_order_relaxed);
	entry.sequence.store(sequence, std::memory_order_release);
	command.setCheckpointNV(reinterpret_cast<const void*>(sequence));
}

void Report(const GraphicContext& graphics, const char* where) {
	if (!Wanted()) return;
	static std::mutex mutex;
	std::lock_guard   lock(mutex);
	Log::WriteToConsoleAndLog(std::string("[gpu-diag] report after ") + where + ", last checkpoint recorded " +
	                          Describe(g_sequence.load()) + "\n");
	ReportQueue(graphics.queue, "main queue");
	ReportQueue(graphics.side_queue, "side queue");
	ReportFault(graphics);
}

} // namespace Libs::Graphics::GpuDiag
