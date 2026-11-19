// Vulkan instance and device creation for the KytyPS5 GPU backend.
// Adapted from KytyPS5 src/graphics/presentation/window/vulkanWindow.cpp (commit 3a7d8030,
// GPL-2.0): the window and its surface belong to AnyPS5's libSceVideoOut, so the device may be
// created before a surface exists, and the headset, DLSS and FSR paths are left out.

#include "glue/VulkanDevice.hpp"
#include "glue/Recorder.hpp"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <string>
#include <vector>

namespace Libs::Graphics {

namespace {

struct SurfaceCapabilities {
	vk::SurfaceCapabilitiesKHR        capabilities {};
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR>   present_modes;
};

} // namespace

struct VulkanExtensions {
	bool enable_validation_layers = false;

	std::vector<const char*>             required_extensions;
	std::vector<vk::ExtensionProperties> available_extensions;
	std::vector<vk::LayerProperties>     available_layers;
};

static constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

static vk::PhysicalDeviceVulkan11Features RequiredVulkan11Features() noexcept {
	return {.storageBuffer16BitAccess = VK_TRUE, .shaderDrawParameters = VK_TRUE};
}

static vk::PhysicalDeviceVulkan12Features RequiredVulkan12Features() noexcept {
	return {.samplerMirrorClampToEdge                     = VK_TRUE,
	        .storageBuffer8BitAccess                      = VK_TRUE,
	        .shaderBufferInt64Atomics                     = VK_TRUE,
	        .shaderSampledImageArrayNonUniformIndexing    = VK_TRUE,
	        .shaderStorageBufferArrayNonUniformIndexing   = VK_TRUE,
	        .shaderStorageImageArrayNonUniformIndexing    = VK_TRUE,
	        .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
	        .descriptorBindingUpdateUnusedWhilePending    = VK_TRUE,
	        .descriptorBindingPartiallyBound              = VK_TRUE,
	        .timelineSemaphore                            = VK_TRUE,
	        .bufferDeviceAddress                          = VK_TRUE,
	        .shaderOutputViewportIndex                    = VK_TRUE,
	        .shaderOutputLayer                            = VK_TRUE};
}

static vk::PhysicalDeviceVulkan13Features RequiredVulkan13Features() noexcept {
	return {.synchronization2 = VK_TRUE, .dynamicRendering = VK_TRUE};
}

static bool HasExtension(const std::vector<vk::ExtensionProperties>& extensions, const char* name) {
	return std::any_of(extensions.begin(), extensions.end(),
	                   [name](const auto& ext) { return strcmp(ext.extensionName, name) == 0; });
}

static bool HasExtension(const std::vector<const char*>& extensions, const char* name) {
	return std::any_of(extensions.begin(), extensions.end(),
	                   [name](const char* ext) { return strcmp(ext, name) == 0; });
}

static bool HasLayer(const std::vector<vk::LayerProperties>& layers, const char* name) {
	return std::any_of(layers.begin(), layers.end(),
	                   [name](const auto& layer) { return strcmp(layer.layerName, name) == 0; });
}

static void GetSurfaceCapabilities(vk::PhysicalDevice physical_device, vk::SurfaceKHR surface,
                                   SurfaceCapabilities& r) {
	RequireVulkanSuccess(physical_device.getSurfaceCapabilitiesKHR(surface, &r.capabilities),
	                     "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

	r.formats = EnumerateVulkan<vk::SurfaceFormatKHR>( // @suppress("Ambiguous problem")
	    "vkGetPhysicalDeviceSurfaceFormatsKHR", [&](uint32_t* count, vk::SurfaceFormatKHR* values) {
		    return physical_device.getSurfaceFormatsKHR(surface, count, values);
	    });
	EXIT_NOT_IMPLEMENTED(r.formats.empty());

	r.present_modes = EnumerateVulkan<vk::PresentModeKHR>( // @suppress("Ambiguous problem")
	    "vkGetPhysicalDeviceSurfacePresentModesKHR",
	    [&](uint32_t* count, vk::PresentModeKHR* values) {
		    return physical_device.getSurfacePresentModesKHR(surface, count, values);
	    });
	EXIT_NOT_IMPLEMENTED(r.present_modes.empty());
}

static bool CheckFormat(vk::PhysicalDevice device, vk::Format format,
                        vk::FormatFeatureFlags features) {
	return (device.getFormatProperties(format).optimalTilingFeatures & features) == features;
}

static uint32_t VulkanFindQueueFamily(vk::PhysicalDevice device, vk::SurfaceKHR surface) {
	EXIT_IF(device == nullptr);

	uint32_t queue_family_count = 0;
	device.getQueueFamilyProperties(&queue_family_count, nullptr);
	std::vector<vk::QueueFamilyProperties> queue_families(queue_family_count);
	device.getQueueFamilyProperties(&queue_family_count, queue_families.data());

	const auto required = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
	for (uint32_t family = 0; family < queue_family_count; family++) {
		const auto& properties             = queue_families[family];
		vk::Bool32  presentation_supported = VK_TRUE;
		if (surface != nullptr) {
			RequireVulkanSuccess(
			    device.getSurfaceSupportKHR(family, surface, &presentation_supported),
			    "vkGetPhysicalDeviceSurfaceSupportKHR");
		}

		LOGF("\tqueue family: %s [count = %u], [present = %s]\n",
		     vk::to_string(properties.queueFlags).c_str(), properties.queueCount,
		     (presentation_supported == VK_TRUE ? "true" : "false"));
		if (properties.queueCount != 0 && (properties.queueFlags & required) == required &&
		    presentation_supported == VK_TRUE) {
			LOGF("\tselected universal queue family %u\n", family);
			return family;
		}
	}
	return static_cast<uint32_t>(-1);
}

// On failure out_device is null and out_rejections says why each device was skipped.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static void VulkanFindPhysicalDevice(vk::Instance instance, vk::SurfaceKHR surface,
                                     const std::vector<const char*>& device_extensions,
                                     vk::PhysicalDevice& out_device, uint32_t& out_queue_family,
                                     std::string& out_rejections) {
	EXIT_IF(instance == nullptr);

	auto devices = EnumerateVulkan<vk::PhysicalDevice>(
	    "vkEnumeratePhysicalDevices", [&](uint32_t* count, vk::PhysicalDevice* values) {
		    return instance.enumeratePhysicalDevices(count, values);
	    });
	EXIT_NOT_IMPLEMENTED(devices.empty());

	if (Config::GetGpuIndex() >= 0) {
		if (static_cast<size_t>(Config::GetGpuIndex()) < devices.size()) {
			devices = {devices[Config::GetGpuIndex()]};
		} else {
			LOGF("Vulkan GPU index %d is unavailable; selecting automatically\n",
			     Config::GetGpuIndex());
		}
	}

	vk::PhysicalDevice best_device       = nullptr;
	uint32_t           best_queue_family = static_cast<uint32_t>(-1);

	for (const auto& device: devices) {
		bool skip_device = false;

		vk::PhysicalDeviceProperties device_properties {};
		device.getProperties(&device_properties);

		const auto reject = [&](const std::string& reason) {
			LOGF("%s\n", reason.c_str());
			out_rejections +=
			    skip_device ? "; " : fmt::format("\n  {}: ", device_properties.deviceName.data());
			out_rejections += reason;
			skip_device = true;
		};

		LOGF("Vulkan device: %s\n", device_properties.deviceName.data());
		if (device_properties.apiVersion < VULKAN_TARGET_API_VERSION) {
			reject(fmt::format("Vulkan {}.{} is required, but device supports only {}.{}.{}",
			                   VK_VERSION_MAJOR(VULKAN_TARGET_API_VERSION),
			                   VK_VERSION_MINOR(VULKAN_TARGET_API_VERSION),
			                   VK_VERSION_MAJOR(device_properties.apiVersion),
			                   VK_VERSION_MINOR(device_properties.apiVersion),
			                   VK_VERSION_PATCH(device_properties.apiVersion)));
			continue;
		}

		vk::PhysicalDeviceImageViewMinLodFeaturesEXT  min_lod {};
		vk::PhysicalDeviceDepthClipEnableFeaturesEXT  clip_enable {.pNext = &min_lod};
		vk::PhysicalDeviceDepthClipControlFeaturesEXT clip_control {.pNext = &clip_enable};
		vk::PhysicalDeviceRobustness2FeaturesEXT      robustness2 {.pNext = &clip_control};
		vk::PhysicalDeviceVulkan11Features            features11 {.pNext = &robustness2};
		vk::PhysicalDeviceVulkan12Features            features12 {.pNext = &features11};
		vk::PhysicalDeviceVulkan13Features            features13 {.pNext = &features12};
		vk::PhysicalDeviceFeatures2                   device_features2 {.pNext = &features13};
		device.getFeatures2(&device_features2);

		const auto queue_family = VulkanFindQueueFamily(device, surface);
		if (queue_family == static_cast<uint32_t>(-1)) {
			reject("No universal graphics, compute, and presentation queue");
		}

		const auto check_feature = [&](vk::Bool32 supported, const char* name) {
			if (supported != VK_TRUE) {
				reject(fmt::format("{} is not supported", name));
			}
		};

		check_feature(min_lod.minLod, "image view minLod");

		check_feature(clip_control.depthClipControl, "depthClipControl");
#if defined(__APPLE__)
		if (clip_enable.depthClipEnable != VK_TRUE) {
			LOGF("depthClipEnable is not supported\n");
		}
#else
		check_feature(clip_enable.depthClipEnable, "depthClipEnable");
#endif
#if !defined(__APPLE__)
		check_feature(device_features2.features.depthClamp, "depthClamp");
#endif

		check_feature(features12.samplerMirrorClampToEdge, "samplerMirrorClampToEdge");
		check_feature(features11.storageBuffer16BitAccess, "storageBuffer16BitAccess");
		check_feature(features11.shaderDrawParameters, "shaderDrawParameters");
		check_feature(features12.storageBuffer8BitAccess, "storageBuffer8BitAccess");
		check_feature(features12.timelineSemaphore, "timelineSemaphore");
		check_feature(features12.shaderOutputLayer, "shaderOutputLayer");
		check_feature(features12.shaderOutputViewportIndex, "shaderOutputViewportIndex");
		check_feature(features12.bufferDeviceAddress, "bufferDeviceAddress");
		check_feature(features12.shaderBufferInt64Atomics, "shaderBufferInt64Atomics");
		check_feature(features12.shaderSampledImageArrayNonUniformIndexing,
		              "shaderSampledImageArrayNonUniformIndexing");
		check_feature(features12.shaderStorageImageArrayNonUniformIndexing,
		              "shaderStorageImageArrayNonUniformIndexing");
		check_feature(features12.shaderStorageBufferArrayNonUniformIndexing,
		              "shaderStorageBufferArrayNonUniformIndexing");
		check_feature(features12.descriptorBindingSampledImageUpdateAfterBind,
		              "descriptorBindingSampledImageUpdateAfterBind");
		check_feature(features12.descriptorBindingUpdateUnusedWhilePending,
		              "descriptorBindingUpdateUnusedWhilePending");
		check_feature(features12.descriptorBindingPartiallyBound,
		              "descriptorBindingPartiallyBound");
		check_feature(features13.robustImageAccess, "robustImageAccess");
		check_feature(robustness2.nullDescriptor, "nullDescriptor");
		check_feature(features13.dynamicRendering, "dynamicRendering");
		check_feature(features13.synchronization2, "synchronization2");
		check_feature(device_features2.features.sampleRateShading, "sampleRateShading");
		check_feature(device_features2.features.depthBiasClamp, "depthBiasClamp");
		check_feature(device_features2.features.shaderClipDistance, "shaderClipDistance");
		check_feature(device_features2.features.shaderCullDistance, "shaderCullDistance");
		check_feature(device_features2.features.largePoints, "largePoints");
		check_feature(device_features2.features.multiViewport, "multiViewport");
		check_feature(device_features2.features.fillModeNonSolid, "fillModeNonSolid");
		check_feature(device_features2.features.fragmentStoresAndAtomics,
		              "fragmentStoresAndAtomics");

		check_feature(device_features2.features.samplerAnisotropy, "samplerAnisotropy");
		check_feature(device_features2.features.robustBufferAccess, "robustBufferAccess");
#if defined(__APPLE__)
		if (device_features2.features.depthBounds != VK_TRUE) {
			LOGF("depthBounds is not supported\n");
		}
#else
		check_feature(device_features2.features.depthBounds, "depthBounds");
#endif
		check_feature(device_features2.features.shaderStorageImageWriteWithoutFormat,
		              "shaderStorageImageWriteWithoutFormat");

		check_feature(device_features2.features.shaderImageGatherExtended,
		              "shaderImageGatherExtended");
		check_feature(device_features2.features.shaderResourceMinLod, "shaderResourceMinLod");

		check_feature(device_features2.features.independentBlend, "independentBlend");
		check_feature(device_features2.features.tessellationShader, "tessellationShader");
		check_feature(device_features2.features.drawIndirectFirstInstance,
		              "drawIndirectFirstInstance");
		check_feature(device_features2.features.multiDrawIndirect, "multiDrawIndirect");

		if (!skip_device) {
			auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
			    "vkEnumerateDeviceExtensionProperties",
			    [&](uint32_t* count, vk::ExtensionProperties* values) {
				    return device.enumerateDeviceExtensionProperties(nullptr, count, values);
			    });
			EXIT_NOT_IMPLEMENTED(available_extensions.empty());

			for (const char* ext: device_extensions) {
				if (!HasExtension(available_extensions, ext)) {
					reject(fmt::format("{} is not supported", ext));
				}
			}

			if (skip_device) {
				for (const auto& ext: available_extensions) {
					LOGF("Vulkan available extension: %s, version = %u\n", ext.extensionName.data(),
					     ext.specVersion);
				}
			}
		}

		SurfaceCapabilities candidate_capabilities;
		if (!skip_device && surface != nullptr) {
			GetSurfaceCapabilities(device, surface, candidate_capabilities);

			if (!(candidate_capabilities.capabilities.supportedUsageFlags &
			      vk::ImageUsageFlagBits::eTransferDst)) {
				reject("Surface cannot be destination of blit");
			}
		}

		const auto check_format = [&](vk::Format format, vk::FormatFeatureFlags features,
		                              const char* usage) {
			if (!skip_device && !CheckFormat(device, format, features)) {
				reject(fmt::format("Format vk::Format::e{} cannot be used as {}",
				                   vk::to_string(format), usage));
			}
		};

		const auto depth_features = vk::FormatFeatureFlagBits::eDepthStencilAttachment;
		const auto texture_features =
		    vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferDst;
		const auto storage_features =
		    vk::FormatFeatureFlagBits::eStorageImage | vk::FormatFeatureFlagBits::eTransferDst;

		check_format(vk::Format::eD32Sfloat, depth_features, "depth buffer");
		check_format(vk::Format::eD32SfloatS8Uint, depth_features, "depth buffer");
		check_format(vk::Format::eD16Unorm, depth_features, "depth buffer");
		check_format(vk::Format::eBc3SrgbBlock, texture_features, "texture");
		check_format(vk::Format::eR8G8B8A8Srgb, texture_features, "texture");
		check_format(vk::Format::eR8Unorm, texture_features, "texture");
		check_format(vk::Format::eR8G8Unorm, texture_features, "texture");

		for (const auto [srgb, unorm]:
		     {std::pair {vk::Format::eR8G8B8A8Srgb, vk::Format::eR8G8B8A8Unorm},
		      std::pair {vk::Format::eB8G8R8A8Srgb, vk::Format::eB8G8R8A8Unorm}}) {
			if (!skip_device && !CheckFormat(device, srgb, storage_features)) {
				LOGF("Format vk::Format::e%s cannot be used as storage image\n",
				     vk::to_string(srgb).c_str());
				check_format(unorm, storage_features, "storage image");
			}
		}

		if (!skip_device && device_properties.limits.maxSamplerAnisotropy < 16.0f) {
			reject("maxSamplerAnisotropy < 16.0f");
		}

		if (skip_device) {
			continue;
		}

		if (best_device == nullptr ||
		    device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
			best_device       = device;
			best_queue_family = queue_family;
		}
	}

	out_device       = best_device;
	out_queue_family = best_queue_family;
}

static vk::Device VulkanCreateDevice(GraphicContext&          graphics,
                                     std::vector<const char*> device_extensions) {
	const auto physical_device = graphics.physical_device;
	const auto queue_family    = graphics.queue_family;
	EXIT_IF(physical_device == nullptr);
	EXIT_IF(queue_family == static_cast<uint32_t>(-1));

	const auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
	    "vkEnumerateDeviceExtensionProperties",
	    [&](uint32_t* count, vk::ExtensionProperties* values) {
		    return physical_device.enumerateDeviceExtensionProperties(nullptr, count, values);
	    });
	// Enables an optional extension the device has.
	const auto add = [&](const char* name) {
		const bool available = HasExtension(available_extensions, name);
		if (available) device_extensions.push_back(name);
		return available;
	};
	// APS5_KYTY_GPU_DIAG: checkpoints, driver diagnostics and fault reports for device losses.
	const bool gpu_diag = std::getenv("APS5_KYTY_GPU_DIAG") != nullptr;
	if (gpu_diag) {
		graphics.diagnostic_checkpoints_enabled = add(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
		graphics.device_fault_enabled            = add(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
	}
	const bool diagnostics_config = gpu_diag && add(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME);
	// Optional feature structs stay in the query only with their extension.
	vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
	                   vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceRobustness2FeaturesEXT,
	                   vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR,
	                   vk::PhysicalDeviceMeshShaderFeaturesEXT,
	                   vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT,
	                   vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT,
	                   vk::PhysicalDeviceProvokingVertexFeaturesEXT,
	                   vk::PhysicalDeviceMaintenance8FeaturesKHR,
	                   vk::PhysicalDeviceShaderClockFeaturesKHR,
	                   vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT,
	                   vk::PhysicalDeviceConditionalRenderingFeaturesEXT,
	                   vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR,
	                   vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR,
	                   vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
	                   vk::PhysicalDeviceRayQueryFeaturesKHR,
	                   vk::PhysicalDeviceCooperativeMatrixFeaturesKHR,
	                   vk::PhysicalDeviceShaderFloat8FeaturesEXT>
	    supported;
	auto& [supported2, supported12, supported13, robustness2, workgroup_layout, mesh,
	       feedback_layout, feedback_dynamic, provoking_vertex, maintenance8, shader_clock,
	       image_atomic_int64, conditional_rendering, barycentric, compute_derivatives,
	       acceleration_structure, ray_query, cooperative_matrix, float8] = supported;
	graphics.memory_budget_ext_enabled = add(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
	// GPU-SRT reads guest memory through imported host pages (BufferCache host view).
	graphics.external_memory_host_enabled = add(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
	add(VK_EXT_DEPTH_RANGE_UNRESTRICTED_EXTENSION_NAME);
	const bool extended_dynamic_state = add(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);
	if (!add(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
	}
	if (!add(VK_EXT_MESH_SHADER_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceMeshShaderFeaturesEXT>();
	}
	// Feedback loops take both extensions or neither.
	if (!HasExtension(available_extensions,
	                  VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) ||
	    !add(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME) ||
	    !add(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT>();
		supported.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT>();
	}
	if (!add(VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceProvokingVertexFeaturesEXT>();
	}
	if (!add(VK_KHR_MAINTENANCE_8_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceMaintenance8FeaturesKHR>();
	}
	if (!add(VK_KHR_SHADER_CLOCK_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceShaderClockFeaturesKHR>();
	}
	if (!add(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT>();
	}
	if (!add(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceConditionalRenderingFeaturesEXT>();
	}
	if (!add(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>();
	}
	if (!add(VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR>();
	}
	// Ray queries take all three extensions or none.
	const bool ray_query_extensions =
	    Config::NativeRtEnabled() &&
	    HasExtension(available_extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) &&
	    HasExtension(available_extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
	    HasExtension(available_extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME);
	if (ray_query_extensions) {
		add(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
		add(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
		add(VK_KHR_RAY_QUERY_EXTENSION_NAME);
	} else {
		supported.unlink<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();
		supported.unlink<vk::PhysicalDeviceRayQueryFeaturesKHR>();
	}
	// FSR's ML upscalers run on cooperative matrices; the backend has no FSR.
	const bool fsr = false;
	if (!fsr || !add(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>();
	}
	if (!fsr || !add(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceShaderFloat8FeaturesEXT>();
	}
	physical_device.getFeatures2(&supported2);
	const bool barycentric_supported = barycentric.fragmentShaderBarycentric == VK_TRUE;
	const bool ray_query_enabled =
	    acceleration_structure.accelerationStructure && ray_query.rayQuery;
	if (!barycentric_supported) {
		Log::WriteToConsoleAndLog(
		    "WARNING: Your GPU is not fully supported (fragmentShaderBarycentric is unavailable). "
		    "Some games may render incorrectly or crash.\n");
	}
	graphics.shader_image_int64_atomics_enabled = image_atomic_int64.shaderImageInt64Atomics;
	graphics.sampler_filter_minmax_enabled      = supported12.samplerFilterMinmax;

	auto features12                     = RequiredVulkan12Features();
	features12.shaderSharedInt64Atomics = supported12.shaderSharedInt64Atomics;
	features12.samplerFilterMinmax      = supported12.samplerFilterMinmax;
	if (!features12.shaderSharedInt64Atomics || !workgroup_layout.workgroupMemoryExplicitLayout) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "WARNING: Native 64-bit LDS atomics are unavailable: shaderSharedInt64Atomics={}, "
		    "workgroupMemoryExplicitLayout={}. Continuing with the selected Vulkan device.\n",
		    features12.shaderSharedInt64Atomics != VK_FALSE,
		    workgroup_layout.workgroupMemoryExplicitLayout != VK_FALSE));
	}
	// Indirect mesh draws launch their records' groups from a task shader.
	graphics.mesh_shader_enabled = mesh.meshShader && mesh.taskShader;

	vk::StructureChain<
	    vk::PhysicalDeviceProperties2, vk::PhysicalDeviceFloatControlsProperties,
	    vk::PhysicalDeviceRobustness2PropertiesEXT, vk::PhysicalDeviceVulkan11Properties,
	    vk::PhysicalDeviceSubgroupSizeControlProperties, vk::PhysicalDevicePushDescriptorProperties,
	    vk::PhysicalDeviceMeshShaderPropertiesEXT>
	    properties;
	auto& [properties2, float_controls, robustness2_props, properties11, subgroup_control,
	       push_descriptor, mesh_properties] = properties;
	if (!graphics.mesh_shader_enabled) {
		properties.unlink<vk::PhysicalDeviceMeshShaderPropertiesEXT>();
	}
	physical_device.getProperties2(&properties2);
	EXIT_NOT_IMPLEMENTED(!(properties11.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eVote));
	graphics.physical_device_properties = properties2.properties;
	graphics.max_push_descriptors       = push_descriptor.maxPushDescriptors;
	graphics.mesh_shader_properties     = mesh_properties;
	LOGF("Select device: %s\n", graphics.physical_device_properties.deviceName.data());

	const auto subgroup_stages = subgroup_control.requiredSubgroupSizeStages;
	graphics.subgroup_size     = properties11.subgroupSize;
	graphics.min_subgroup_size = subgroup_control.minSubgroupSize;
	graphics.max_subgroup_size = subgroup_control.maxSubgroupSize;
	graphics.compute_subgroup_size_control_enabled =
	    supported13.subgroupSizeControl == VK_TRUE &&
	    (subgroup_stages & vk::ShaderStageFlagBits::eCompute) &&
	    subgroup_control.minSubgroupSize <= 64 && subgroup_control.maxSubgroupSize >= 64;

	LOGF("Vulkan subgroup: default=%u min=%u max=%u stages=0x%08x size_control=%s wave64=%s\n",
	     graphics.subgroup_size, graphics.min_subgroup_size, graphics.max_subgroup_size,
	     static_cast<vk::ShaderStageFlags::MaskType>(subgroup_stages),
	     graphics.compute_subgroup_size_control_enabled ? "true" : "false",
	     graphics.SupportsComputeWave64() ? "true" : "false");
	graphics.provoking_vertex_last_enabled    = provoking_vertex.provokingVertexLast;
	graphics.attachment_feedback_loop_enabled = feedback_layout.attachmentFeedbackLoopLayout &&
	                                            feedback_dynamic.attachmentFeedbackLoopDynamicState;
	graphics.maintenance8_enabled             = maintenance8.maintenance8;
	graphics.ray_query_enabled                = ray_query_enabled;
	// A coarser alignment rounds the range up past the guest's last byte.
	graphics.robust_buffer_access_enabled =
	    robustness2.robustBufferAccess2 &&
	    robustness2_props.robustStorageBufferAccessSizeAlignment == 1u;
	graphics.r8_srgb_views   = graphics.SamplesFiltered(vk::Format::eR8Srgb);
	graphics.r8g8_srgb_views = graphics.SamplesFiltered(vk::Format::eR8G8Srgb);
	constexpr auto blended_target = vk::FormatFeatureFlagBits::eColorAttachment |
	                                vk::FormatFeatureFlagBits::eColorAttachmentBlend;
	graphics.r4g4b4a4_render_targets =
	    (graphics.GetFormatProperties(vk::Format::eR4G4B4A4UnormPack16).optimalTilingFeatures &
	     blended_target) == blended_target;
	// The device clock ticks at timestampPeriod nanoseconds.
	if (shader_clock.shaderDeviceClock) {
		graphics.device_clock_period_ps = static_cast<uint32_t>(
		    std::lround(graphics.physical_device_properties.limits.timestampPeriod * 1000.0f));
	}
	graphics.conditional_rendering_enabled = conditional_rendering.conditionalRendering;
	LOGF("Vulkan maintenance8 support: %s\n",
	     graphics.maintenance8_enabled ? "true" : "false");
	LOGF("Vulkan depth feedback support: %s\n",
	     graphics.attachment_feedback_loop_enabled ? "true" : "false");
	if (graphics.mesh_shader_enabled) {
		LOGF("Vulkan MeshEXT: invocations=%u vertices=%u primitives=%u shared=%u\n",
		     graphics.mesh_shader_properties.maxMeshWorkGroupInvocations,
		     graphics.mesh_shader_properties.maxMeshOutputVertices,
		     graphics.mesh_shader_properties.maxMeshOutputPrimitives,
		     graphics.mesh_shader_properties.maxMeshSharedMemorySize);
	}
	// Device selection checks the other required features.
	EXIT_NOT_IMPLEMENTED(supported2.features.shaderInt64 != VK_TRUE);
	EXIT_NOT_IMPLEMENTED(supported2.features.vertexPipelineStoresAndAtomics != VK_TRUE);
	EXIT_NOT_IMPLEMENTED(supported2.features.dualSrcBlend != VK_TRUE);
	vk::PhysicalDeviceFeatures device_features {};
	device_features.fragmentStoresAndAtomics = VK_TRUE;
	device_features.samplerAnisotropy        = VK_TRUE;
	device_features.robustBufferAccess       = VK_TRUE;
#if !defined(__APPLE__)
	device_features.depthBounds = VK_TRUE; // unsupported by MoltenVK
	device_features.depthClamp  = VK_TRUE;
#endif
	device_features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
	device_features.shaderStorageImageMultisample =
	    supported2.features.shaderStorageImageMultisample;
	device_features.shaderImageGatherExtended            = VK_TRUE;
	device_features.shaderResourceMinLod                 = VK_TRUE;
	device_features.independentBlend                     = VK_TRUE;
	device_features.dualSrcBlend                         = VK_TRUE;
	device_features.tessellationShader                   = VK_TRUE;
	device_features.drawIndirectFirstInstance            = VK_TRUE;
	device_features.multiDrawIndirect                    = VK_TRUE;
	device_features.sampleRateShading                    = VK_TRUE;
	device_features.depthBiasClamp                       = VK_TRUE;
	device_features.shaderClipDistance                   = VK_TRUE;
	device_features.shaderCullDistance                   = VK_TRUE;
	device_features.largePoints                          = VK_TRUE;
	device_features.multiViewport                        = VK_TRUE;
	device_features.fillModeNonSolid                      = VK_TRUE;
	device_features.vertexPipelineStoresAndAtomics       = VK_TRUE;
	// Storage image sample counts beyond 1x require the feature.
	if (!device_features.shaderStorageImageMultisample) {
		graphics.physical_device_properties.limits.storageImageSampleCounts =
		    vk::SampleCountFlagBits::e1;
	}
	// Occlusion queries reset their queries from the host and count samples exactly where they can.
	graphics.occlusion_queries_enabled =
	    Config::OcclusionQueriesEnabled() && supported12.hostQueryReset;
	graphics.occlusion_query_precise_enabled =
	    graphics.occlusion_queries_enabled && supported2.features.occlusionQueryPrecise;
	// APS5_KYTY_GPUTIME: timestamps around draws and dispatches, reset from the host as well.
	{
		uint32_t family_count = 0;
		graphics.physical_device.getQueueFamilyProperties(&family_count, nullptr);
		std::vector<vk::QueueFamilyProperties> families(family_count);
		graphics.physical_device.getQueueFamilyProperties(&family_count, families.data());
		graphics.timestamp_valid_bits =
		    graphics.queue_family < family_count ? families[graphics.queue_family].timestampValidBits : 0;
	}
	graphics.gpu_time_enabled = std::getenv("APS5_KYTY_GPUTIME") != nullptr && supported12.hostQueryReset &&
	                            graphics.timestamp_valid_bits != 0;
	features12.hostQueryReset             = graphics.occlusion_queries_enabled || graphics.gpu_time_enabled;
	device_features.occlusionQueryPrecise = graphics.occlusion_query_precise_enabled;
	LOGF("Vulkan ray query: %s\n", ray_query_enabled ? "true" : "false");
	LOGF("Vulkan occlusion queries: %s, precise: %s\n",
	     graphics.occlusion_queries_enabled ? "true" : "false",
	     graphics.occlusion_query_precise_enabled ? "true" : "false");
	device_features.shaderInt64 = VK_TRUE;
	device_features.shaderFloat64 =
	    supported2.features.shaderFloat64 && float_controls.shaderSignedZeroInfNanPreserveFloat64;

	auto features13                = RequiredVulkan13Features();
	features13.robustImageAccess   = supported13.robustImageAccess;
	features13.subgroupSizeControl = graphics.compute_subgroup_size_control_enabled;
	if (fsr) {
		device_features.shaderInt16            = supported2.features.shaderInt16;
		features12.shaderFloat16               = supported12.shaderFloat16;
		features12.shaderInt8                  = supported12.shaderInt8;
		features12.separateDepthStencilLayouts = supported12.separateDepthStencilLayouts;
		features13.shaderIntegerDotProduct     = supported13.shaderIntegerDotProduct;
		features13.computeFullSubgroups        = supported13.computeFullSubgroups;
	}

	LOGF("Vulkan robustness: robustImageAccess=%s robustImageAccess2=%s buffer bounds=%s\n",
	     features13.robustImageAccess == VK_TRUE ? "true" : "false",
	     robustness2.robustImageAccess2 == VK_TRUE ? "true" : "false",
	     graphics.robust_buffer_access_enabled ? "robustBufferAccess2" : "shader checks");

	// Two queues of the family when it has them: the second runs early compute.
	uint32_t family_count = 0;
	physical_device.getQueueFamilyProperties(&family_count, nullptr);
	std::vector<vk::QueueFamilyProperties> families(family_count);
	physical_device.getQueueFamilyProperties(&family_count, families.data());
	const uint32_t queue_count =
	    queue_family < family_count && families[queue_family].queueCount >= 2 ? 2u : 1u;
	const float                     queue_priorities[2] = {1.0f, 1.0f};
	const vk::DeviceQueueCreateInfo queue_create_info {
	    .queueFamilyIndex = queue_family, .queueCount = queue_count, .pQueuePriorities = queue_priorities};
	graphics.side_queue_count = queue_count;
	const auto                 extension_count = static_cast<uint32_t>(device_extensions.size());
	const vk::DeviceCreateInfo create_info {.queueCreateInfoCount    = 1,
	                                        .pQueueCreateInfos       = &queue_create_info,
	                                        .enabledExtensionCount   = extension_count,
	                                        .ppEnabledExtensionNames = device_extensions.data(),
	                                        .pEnabledFeatures        = &device_features};
	provoking_vertex.transformFeedbackPreservesProvokingVertex = VK_FALSE;
	image_atomic_int64.sparseImageInt64Atomics                 = VK_FALSE;
	// Robustness2 enables what the device has; selection requires nullDescriptor.
	vk::StructureChain create {
	    create_info, RequiredVulkan11Features(), features12, features13, robustness2,
	    vk::PhysicalDeviceDepthClipControlFeaturesEXT {.depthClipControl = VK_TRUE},
	    vk::PhysicalDeviceImageViewMinLodFeaturesEXT {.minLod = VK_TRUE},
	    vk::PhysicalDeviceDepthClipEnableFeaturesEXT {.depthClipEnable = VK_TRUE},
	    vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR {
	        .workgroupMemoryExplicitLayout = workgroup_layout.workgroupMemoryExplicitLayout},
	    barycentric,
	    vk::PhysicalDeviceMeshShaderFeaturesEXT {.taskShader = VK_TRUE, .meshShader = VK_TRUE},
	    feedback_layout, feedback_dynamic, provoking_vertex, image_atomic_int64, maintenance8,
	    shader_clock, conditional_rendering, compute_derivatives,
	    vk::PhysicalDeviceAccelerationStructureFeaturesKHR {.accelerationStructure = VK_TRUE},
	    vk::PhysicalDeviceRayQueryFeaturesKHR {.rayQuery = VK_TRUE}, cooperative_matrix, float8,
	    // The renderer only uses the Vulkan 1.3 core dynamic states, but RADV (Mesa 26.3) creates
	    // its PS epilog cache only when this EXT feature is enabled and crashes in vkCmdDraw when
	    // a depth- or stencil-writing fragment shader needs an epilog.
	    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT {.extendedDynamicState = VK_TRUE},
	    vk::PhysicalDeviceFaultFeaturesEXT {.deviceFault = VK_TRUE},
	    vk::DeviceDiagnosticsConfigCreateInfoNV {
	        .flags = vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderDebugInfo |
	                 vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableResourceTracking |
	                 vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableAutomaticCheckpoints}};
	if (!graphics.device_fault_enabled) create.unlink<vk::PhysicalDeviceFaultFeaturesEXT>();
	if (!diagnostics_config) create.unlink<vk::DeviceDiagnosticsConfigCreateInfoNV>();
	if (!workgroup_layout.workgroupMemoryExplicitLayout) {
		create.unlink<vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
	}
#if defined(__APPLE__)
	// MoltenVK lacks these extensions (see CreateVulkan).
	create.unlink<vk::PhysicalDeviceDepthClipEnableFeaturesEXT>();
	create.unlink<vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>();
#else
	if (!barycentric_supported) {
		create.unlink<vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>();
	}
#endif
	if (!graphics.mesh_shader_enabled) create.unlink<vk::PhysicalDeviceMeshShaderFeaturesEXT>();
	if (!graphics.attachment_feedback_loop_enabled) {
		create.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT>();
		create.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT>();
	}
	if (!graphics.provoking_vertex_last_enabled) {
		create.unlink<vk::PhysicalDeviceProvokingVertexFeaturesEXT>();
	}
	if (!graphics.shader_image_int64_atomics_enabled) {
		create.unlink<vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT>();
	}
	if (!graphics.maintenance8_enabled) create.unlink<vk::PhysicalDeviceMaintenance8FeaturesKHR>();
	if (graphics.device_clock_period_ps == 0u) {
		create.unlink<vk::PhysicalDeviceShaderClockFeaturesKHR>();
	}
	if (!graphics.conditional_rendering_enabled) {
		create.unlink<vk::PhysicalDeviceConditionalRenderingFeaturesEXT>();
	}
	if (!compute_derivatives.computeDerivativeGroupQuads) {
		create.unlink<vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR>();
	}
	if (!extended_dynamic_state) {
		create.unlink<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
	}
	if (!cooperative_matrix.cooperativeMatrix) {
		create.unlink<vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>();
	}
	if (!float8.shaderFloat8) create.unlink<vk::PhysicalDeviceShaderFloat8FeaturesEXT>();
	if (!ray_query_enabled) {
		create.unlink<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();
		create.unlink<vk::PhysicalDeviceRayQueryFeaturesKHR>();
	}

	vk::Device device = nullptr;
	RequireVulkanSuccess(physical_device.createDevice(&create.get(), nullptr, &device),
	                     "vkCreateDevice");
	return device;
}

static void VulkanGetExtensions(VulkanExtensions& r, std::span<const char* const> window_extensions) {
	// The surface extensions are always enabled: the window may be attached after the device.
	r.required_extensions = {VK_KHR_SURFACE_EXTENSION_NAME};
#if defined(_WIN32)
	r.required_extensions.push_back("VK_KHR_win32_surface");
#endif
	for (const char* extension: window_extensions) {
		if (!HasExtension(r.required_extensions, extension)) {
			r.required_extensions.push_back(extension);
		}
	}

	r.available_extensions =
	    EnumerateVulkan<vk::ExtensionProperties>( // @suppress("Ambiguous problem")
	        "vkEnumerateInstanceExtensionProperties",
	        [](uint32_t* count, vk::ExtensionProperties* values) {
		        return vk::enumerateInstanceExtensionProperties(nullptr, count, values);
	        });

	r.enable_validation_layers = Config::VulkanValidationEnabled();
	if (Config::HdrEnabled() &&
	    HasExtension(r.available_extensions, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME) &&
	    !HasExtension(r.required_extensions, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME)) {
		r.required_extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
	}

	if (HasExtension(r.available_extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
		r.required_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	} else {
		r.enable_validation_layers = false;
	}

	for (const char* ext: r.required_extensions) {
		LOGF("Vulkan required extension: %s\n", ext);
	}

	for (const auto& ext: r.available_extensions) {
		LOGF("Vulkan available extension: %s, version = %u\n", ext.extensionName.data(),
		     ext.specVersion);
	}

	r.available_layers = EnumerateVulkan<vk::LayerProperties>( // @suppress("Ambiguous problem")
	    "vkEnumerateInstanceLayerProperties", [](uint32_t* count, vk::LayerProperties* values) {
		    return vk::enumerateInstanceLayerProperties(count, values);
	    });

	for (const auto& l: r.available_layers) {
		LOGF("Vulkan available layer: %s, specVersion = %u, implVersion = %u, %s\n",
		     l.layerName.data(), l.specVersion, l.implementationVersion, l.description.data());
	}

	if (r.enable_validation_layers && !HasLayer(r.available_layers, kValidationLayer)) {
		LOGF("no validation layer: %s\n", kValidationLayer);
		r.enable_validation_layers = false;
	}

	if (r.enable_validation_layers) {
		auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
		    "vkEnumerateInstanceExtensionProperties",
		    [](uint32_t* count, vk::ExtensionProperties* values) {
			    return vk::enumerateInstanceExtensionProperties(kValidationLayer, count, values);
		    });

		for (const auto& ext: available_extensions) {
			LOGF("VK_LAYER_KHRONOS_validation available extension: %s, version = %u\n",
			     ext.extensionName.data(), ext.specVersion);
		}

		if (HasExtension(available_extensions, VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME)) {
			r.required_extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
		} else {
			r.enable_validation_layers = false;
		}
	}
}

static VKAPI_ATTR vk::Bool32 VKAPI_CALL VulkanDebugMessengerCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT      message_severity,
    vk::DebugUtilsMessageTypeFlagsEXT             message_types,
    const vk::DebugUtilsMessengerCallbackDataEXT* callback_data, void* /*user_data*/) {
	EXIT_IF(callback_data == nullptr);
	EXIT_IF(callback_data->pMessage == nullptr);

	const char*     severity_str   = nullptr;
	fmt::text_style severity_style = Log::Color::Default;
	bool            skip           = false;
	bool            error          = false;
	bool            debug_printf   = false;
	switch (message_severity) {
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
			severity_str   = "V";
			severity_style = Log::Color::BrightWhite;
			skip           = true;
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
			if ((message_types & vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation) &&
			    Config::SpirvDebugPrintfEnabled() && callback_data->pMessageIdName != nullptr &&
			    strcmp(callback_data->pMessageIdName, "UNASSIGNED-DEBUG-PRINTF") == 0) {
				debug_printf   = true;
				severity_style = Log::Color::BrightYellow;
				skip           = true;
			} else {
				severity_str   = "I";
				severity_style = Log::Color::Default;
				skip           = true;
			}
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
			severity_str   = "W";
			severity_style = Log::Color::Red;
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
			severity_str   = "E";
			severity_style = Log::Color::BrightRed;
			// Only validation errors are fatal; GENERAL-type errors can come
			// from unrelated loader/layer issues (e.g. a broken overlay).
			error = static_cast<bool>(message_types &
			                          vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
			break;
		default: severity_str = "?";
	}

	// AnyPS5: validation errors are logged (APS5_KYTY_FATAL_VALIDATION=1 makes them fatal) so a
	// diagnostic run reaches the failure it is looking for.
	static const bool fatal = std::getenv("APS5_KYTY_FATAL_VALIDATION") != nullptr;
	if (error && fatal) {
		EXIT("[Vulkan][%s][%u]: %s\n", severity_str, static_cast<uint32_t>(message_types),
		     callback_data->pMessage);
	}

	if (!skip) {
		LOGF_COLOR(severity_style, "[Vulkan][%s][%u]: %s\n", severity_str,
		           static_cast<uint32_t>(message_types), callback_data->pMessage);
	}

	if (debug_printf) {
		auto strs = Common::Split(std::string(callback_data->pMessage), '|');
		if (!strs.empty()) {
			LOGF_COLOR(severity_style, "%s\n", strs[strs.size() - 1].c_str());
		}
	}

	return VK_FALSE;
}

static void VulkanCheckInstanceVersion() {
	uint32_t version = VK_API_VERSION_1_0;
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkEnumerateInstanceVersion != nullptr) {
		RequireVulkanSuccess(vk::enumerateInstanceVersion(&version), "vkEnumerateInstanceVersion");
	}
	if (version < VULKAN_TARGET_API_VERSION) {
		EXIT("Vulkan %u.%u is required, but the loader supports only %u.%u.%u\n",
		     VK_VERSION_MAJOR(VULKAN_TARGET_API_VERSION),
		     VK_VERSION_MINOR(VULKAN_TARGET_API_VERSION), VK_VERSION_MAJOR(version),
		     VK_VERSION_MINOR(version), VK_VERSION_PATCH(version));
	}
}

void CreateBackendInstance(GraphicContext& graphics, PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                           std::span<const char* const> window_extensions) {
	EXIT_IF(graphics.instance != nullptr || get_instance_proc_addr == nullptr);
	VULKAN_HPP_DEFAULT_DISPATCHER.init(get_instance_proc_addr);

	VulkanExtensions r;
	VulkanGetExtensions(r, window_extensions);
	VulkanCheckInstanceVersion();

	const vk::ApplicationInfo app_info {.pApplicationName   = "AnyPS5",
	                                    .applicationVersion = 1,
	                                    .pEngineName        = "AnyPS5 KytyPS5 backend",
	                                    .engineVersion      = 1,
	                                    .apiVersion         = VULKAN_TARGET_API_VERSION};

	vk::DebugUtilsMessengerCreateInfoEXT dbg_create_info {};
	dbg_create_info.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
	                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
	dbg_create_info.messageType     = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
	                                  vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
	                                  vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
	dbg_create_info.pfnUserCallback = VulkanDebugMessengerCallback;

	vk::InstanceCreateInfo inst_info {};
	inst_info.pNext                   = r.enable_validation_layers ? &dbg_create_info : nullptr;
	inst_info.pApplicationInfo        = &app_info;
	inst_info.enabledExtensionCount   = static_cast<uint32_t>(r.required_extensions.size());
	inst_info.ppEnabledExtensionNames = r.required_extensions.data();
	inst_info.enabledLayerCount       = r.enable_validation_layers ? 1u : 0u;
	inst_info.ppEnabledLayerNames     = &kValidationLayer;

	const vk::Result result = vk::createInstance(&inst_info, nullptr, &graphics.instance);
	if (result != vk::Result::eSuccess) {
		EXIT("Could not create a Vulkan instance: %s\n", vk::to_string(result).c_str());
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(graphics.instance);
	if (r.enable_validation_layers &&
	    graphics.instance.createDebugUtilsMessengerEXT(&dbg_create_info, nullptr,
	                                                   &graphics.debug_messenger) !=
	        vk::Result::eSuccess) {
		graphics.debug_messenger = nullptr;
	}
}

void CreateBackendDevice(GraphicContext& graphics, vk::SurfaceKHR surface) {
	EXIT_IF(graphics.instance == nullptr || graphics.device != nullptr);
	std::vector<const char*> device_extensions = {
	    VK_KHR_SWAPCHAIN_EXTENSION_NAME,          VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME,
	    VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
	    VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,       "VK_KHR_maintenance1",
	    VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME};

	std::string rejected_devices;
	VulkanFindPhysicalDevice(graphics.instance, surface, device_extensions,
	                         graphics.physical_device, graphics.queue_family, rejected_devices);
	if (graphics.physical_device == nullptr) {
		EXIT("Could not find a suitable Vulkan device:%s", rejected_devices.c_str());
	}

	graphics.physical_device.getMemoryProperties(&graphics.physical_device_memory_properties);
	const vk::PhysicalDeviceImageFormatInfo2 block_texel_view_info {
	    .format = vk::Format::eBc1RgbaUnormBlock,
	    .type   = vk::ImageType::e2D,
	    .tiling = vk::ImageTiling::eOptimal,
	    .usage  = vk::ImageUsageFlagBits::eSampled,
	    .flags  = vk::ImageCreateFlagBits::eBlockTexelViewCompatible,
	};
	const auto block_texel_view_props =
	    graphics.physical_device.getImageFormatProperties2(block_texel_view_info);
	graphics.supports_block_texel_view = block_texel_view_props.result == vk::Result::eSuccess;

	graphics.device = VulkanCreateDevice(graphics, std::move(device_extensions));
	VULKAN_HPP_DEFAULT_DISPATCHER.init(graphics.device);
	AgcDriver::VramBackend::Recorder::Install();
	graphics.device.getQueue(graphics.queue_family, 0, &graphics.queue);
	EXIT_IF(graphics.queue == nullptr);
	if (graphics.side_queue_count >= 2) {
		graphics.device.getQueue(graphics.queue_family, 1, &graphics.side_queue);
	}
	if (!graphics.CreateAllocator()) {
		EXIT("Could not create the Vulkan memory allocator\n");
	}
}

bool SurfaceSupportsPresent(const GraphicContext& graphics, vk::SurfaceKHR surface) {
	vk::Bool32 supported = VK_FALSE;
	return graphics.physical_device.getSurfaceSupportKHR(graphics.queue_family, surface,
	                                                     &supported) == vk::Result::eSuccess &&
	       supported == VK_TRUE;
}

void DestroyBackendDevice(GraphicContext& graphics) {
	if (graphics.device != nullptr) {
		RequireVulkanSuccess(graphics.device.waitIdle(), "wait for Vulkan device shutdown");
		graphics.DestroyAllocator();
		graphics.device.destroy(nullptr);
		graphics.device = nullptr;
		graphics.queue  = nullptr;
	}
	if (graphics.debug_messenger != nullptr) {
		graphics.instance.destroyDebugUtilsMessengerEXT(graphics.debug_messenger, nullptr);
		graphics.debug_messenger = nullptr;
	}
	if (graphics.instance != nullptr) {
		graphics.instance.destroy(nullptr);
		graphics.instance        = nullptr;
		graphics.physical_device = nullptr;
	}
}

} // namespace Libs::Graphics
