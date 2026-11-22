#include "common/emulatorConfig.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Settings of the KytyPS5 GPU backend. Defaults follow the KytyPS5 test runs
// (shader cache on, read-backs and tessellation off); APS5_KYTY_<NAME>=0/1 overrides.

namespace Config {

namespace {

bool Flag(const char* name, bool fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') {
		return fallback;
	}
	return std::strcmp(value, "0") != 0;
}

uint32_t Number(const char* name, uint32_t fallback) {
	const char* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? static_cast<uint32_t>(std::strtoul(value, nullptr, 0))
	                                          : fallback;
}

std::filesystem::path Path(const char* name, const char* fallback) {
	const char* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? std::filesystem::path(value) : std::filesystem::path(fallback);
}

} // namespace

uint32_t GetScreenWidth() {
	return Number("APS5_KYTY_SCREEN_WIDTH", 1920);
}

uint32_t GetScreenHeight() {
	return Number("APS5_KYTY_SCREEN_HEIGHT", 1080);
}

PresentMode GetPresentMode() {
	return PresentMode::Mailbox;
}

int32_t GetGpuIndex() {
	const char* value = std::getenv("APS5_KYTY_GPU_INDEX");
	return value != nullptr && *value != '\0' ? static_cast<int32_t>(std::strtol(value, nullptr, 0)) : -1;
}

bool HdrEnabled() {
	return false;
}

bool Ps5ProEnabled() {
	return false;
}

PssrUpscaler GetPssrUpscaler() {
	return PssrUpscaler::Linear;
}

const std::filesystem::path& GetDlssFolder() {
	static const std::filesystem::path empty;
	return empty;
}

uint32_t GetDlssPreset() {
	return 0;
}

const std::filesystem::path& GetFsrFolder() {
	static const std::filesystem::path empty;
	return empty;
}

uint32_t GetVblankFrequency() {
	return 60;
}

bool VulkanValidationEnabled() {
	static const bool enabled = Flag("APS5_KYTY_VALIDATION", false);
	return enabled;
}

bool ShaderValidationEnabled() {
	static const bool enabled = Flag("APS5_KYTY_SHADER_VALIDATION", false);
	return enabled;
}

bool ShaderCacheEnabled() {
	static const bool enabled = Flag("APS5_KYTY_SHADER_CACHE", true);
	return enabled;
}

LogDirection GetShaderLogDirection() {
	static const bool enabled = Flag("APS5_KYTY_SHADER_LOG", false);
	return enabled ? LogDirection::File : LogDirection::Silent;
}

bool ShaderLogWanted(uint64_t hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> list;
		const char*           text = std::getenv("APS5_KYTY_SHADER_LOG_HASHES");
		while (text != nullptr && *text != 0) {
			char*      end   = nullptr;
			const auto value = std::strtoull(text, &end, 16);
			if (end == text) break;
			list.push_back(value);
			text = *end == ',' ? end + 1 : end;
		}
		return list;
	}();
	return hashes.empty() || std::find(hashes.begin(), hashes.end(), hash) != hashes.end();
}

std::filesystem::path GetShaderLogFolder() {
	return Path("APS5_KYTY_SHADER_LOG_FOLDER", "_Shaders");
}

bool GraphicsDebugDumpEnabled() {
	static const bool enabled = Flag("APS5_KYTY_GRAPHICS_DUMP", false);
	return enabled;
}

LogDirection GetPrintfDirection() {
	return LogDirection::Silent;
}

bool SpirvDebugPrintfEnabled() {
	return false;
}

bool GpuAssistedValidationEnabled() {
	return false;
}

bool ReadbackLinearImagesEnabled() {
	static const bool enabled = Flag("APS5_KYTY_READBACK_LINEAR", false);
	return enabled;
}

bool ReadbackTiledImagesEnabled() {
	static const bool enabled = Flag("APS5_KYTY_READBACK_TILED", false);
	return enabled;
}

bool ReadbackStorageImagesEnabled() {
	static const bool enabled = Flag("APS5_KYTY_READBACK_STORAGE", false);
	return enabled;
}

bool SyncRawImageBuffersEnabled() {
	static const bool enabled = Flag("APS5_KYTY_SYNC_RAW_IMAGE_BUFFERS", false);
	return enabled;
}

uint32_t GetFlushDraws() {
	static const uint32_t draws = Number("APS5_KYTY_FLUSH_DRAWS", 0);
	return draws;
}

bool TessellationEnabled() {
	static const bool enabled = Flag("APS5_KYTY_TESSELLATION", false);
	return enabled;
}

bool OcclusionQueriesEnabled() {
	static const bool enabled = Flag("APS5_KYTY_OCCLUSION_QUERIES", false);
	return enabled;
}

bool RelaxedShaderFloatEnabled() {
	static const bool enabled = Flag("APS5_KYTY_RELAXED_FLOAT", false);
	return enabled;
}

bool NativeRtEnabled() {
	static const bool enabled = Flag("APS5_KYTY_NATIVE_RT", true);
	return enabled;
}

bool LodStatsEnabled() {
	static const bool enabled = Flag("APS5_KYTY_LOD_STATS", false);
	return enabled;
}

} // namespace Config
