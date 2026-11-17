#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <mutex>
#include <vector>

namespace Libs::Graphics {

namespace {

// Backings of deleted images, kept for a moment for a new image with the same parameters: a guest
// surface used alternately as depth and as color gets two images that replace each other every
// frame (GTA V shadow maps), and creating and freeing their memory each time is slow. An image is
// deleted only after the GPU finished with it, so its backing can serve a new image at once.
struct RecycledImage {
	vk::ImageCreateInfo                   info;
	vk::Image                             image      = nullptr;
	VmaAllocation                         allocation = nullptr;
	uint64_t                              size       = 0;
	std::chrono::steady_clock::time_point freed;
};

constexpr uint64_t RecycleBytes = uint64_t {128} << 20u;
constexpr auto     RecycleAge   = std::chrono::seconds(2);

std::mutex                 g_recycle_mutex;
std::vector<RecycledImage> g_recycled;
uint64_t                   g_recycled_bytes = 0;

bool Recyclable(const vk::ImageCreateInfo& info) {
	return info.pNext == nullptr && info.tiling == vk::ImageTiling::eOptimal &&
	       info.initialLayout == vk::ImageLayout::eUndefined &&
	       info.sharingMode == vk::SharingMode::eExclusive;
}

bool SameParameters(const vk::ImageCreateInfo& a, const vk::ImageCreateInfo& b) {
	return a.flags == b.flags && a.imageType == b.imageType && a.format == b.format &&
	       a.extent == b.extent && a.mipLevels == b.mipLevels && a.arrayLayers == b.arrayLayers &&
	       a.samples == b.samples && a.usage == b.usage;
}

// Frees the backings kept too long, then the oldest while the pool is over its size (locked).
void TrimRecycled(VmaAllocator allocator, uint64_t limit) {
	const auto now = std::chrono::steady_clock::now();
	while (!g_recycled.empty() &&
	       (g_recycled_bytes > limit || now - g_recycled.front().freed > RecycleAge)) {
		auto& oldest = g_recycled.front();
		vmaDestroyImage(allocator, oldest.image, oldest.allocation);
		g_recycled_bytes -= oldest.size;
		g_recycled.erase(g_recycled.begin());
	}
}

} // namespace

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	{
		std::lock_guard lock(g_recycle_mutex);
		TrimRecycled(allocator, 0);
		EXIT_IF(!g_recycled.empty());
	}
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = physical_device_memory_properties;
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint32_t GraphicContext::GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto count = physical_device_memory_properties.memoryHeapCount;
	for (uint32_t heap = 0; heap < count; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		heaps[heap] = {.usage        = budgets[heap].usage,
		               .budget       = budgets[heap].budget,
		               .allocations  = budgets[heap].statistics.allocationCount,
		               .device_local = static_cast<bool>(properties.flags &
		                                                 vk::MemoryHeapFlagBits::eDeviceLocal)};
	}
	return count;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

	const bool recyclable = Recyclable(image_info);
	if (recyclable) {
		std::lock_guard lock(g_recycle_mutex);
		TrimRecycled(allocator, RecycleBytes);
		const auto match = std::ranges::find_if(g_recycled, [&](const RecycledImage& entry) {
			return SameParameters(entry.info, image_info);
		});
		if (match != g_recycled.end()) {
			image.image      = match->image;
			image.allocation = match->allocation;
			g_recycled_bytes -= match->size;
			g_recycled.erase(match);
		}
	}
	if (image.image == nullptr) {
		vk::Image::CType native_image = VK_NULL_HANDLE;
		const auto        result       = static_cast<vk::Result>(vmaCreateImage(
		    allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info), &alloc_info,
		    &native_image, &image.allocation, nullptr));
		image.image = native_image;
		if (result != vk::Result::eSuccess) {
			LogMemoryBudget();
			return false;
		}
	}
	image.recyclable = recyclable;

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	if (image.recyclable) {
		vk::ImageCreateInfo info {};
		info.flags       = image.flags;
		info.imageType   = image.image_type;
		info.format      = image.format;
		info.extent      = image.extent;
		info.mipLevels   = image.mip_levels;
		info.arrayLayers = image.layers;
		info.samples     = static_cast<vk::SampleCountFlagBits>(image.samples);
		info.usage       = image.usage;
		VmaAllocationInfo allocation {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation);
		std::lock_guard lock(g_recycle_mutex);
		g_recycled.push_back({info, image.image, image.allocation, allocation.size,
		                      std::chrono::steady_clock::now()});
		g_recycled_bytes += allocation.size;
		TrimRecycled(allocator, RecycleBytes);
	} else {
		vmaDestroyImage(allocator, image.image, image.allocation);
	}
	image.image      = nullptr;
	image.allocation = nullptr;
	image.recyclable = false;
}

} // namespace Libs::Graphics
