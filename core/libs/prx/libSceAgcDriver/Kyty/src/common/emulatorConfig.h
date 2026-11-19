#ifndef KYTY_COMMON_EMULATOR_CONFIG_H_
#define KYTY_COMMON_EMULATOR_CONFIG_H_

// AnyPS5: the settings the KytyPS5 GPU stack reads. KytyPS5 loads them from its command line;
// here they are fixed defaults (those of the KytyPS5 runs title B was measured with) that
// APS5_KYTY_* environment variables can change (glue/Config.cpp).

#include "common/common.h"

#include <cstdint>
#include <filesystem>

namespace Config {

enum class LogDirection { Silent, Console, File };
enum class PresentMode { Fifo, Mailbox, Immediate };
enum class PssrUpscaler { Linear, DLSS, FSR };

uint32_t GetScreenWidth();
uint32_t GetScreenHeight();
PresentMode GetPresentMode();
int32_t GetGpuIndex();
bool HdrEnabled();
bool Ps5ProEnabled();
PssrUpscaler GetPssrUpscaler();
const std::filesystem::path& GetDlssFolder();
uint32_t GetDlssPreset();
const std::filesystem::path& GetFsrFolder();
uint32_t GetVblankFrequency();
bool VulkanValidationEnabled();
bool ShaderValidationEnabled();
bool ShaderCacheEnabled();
LogDirection GetShaderLogDirection();
std::filesystem::path GetShaderLogFolder();
// APS5_KYTY_SHADER_LOG_HASHES: comma-separated hexadecimal shader hashes the shader log keeps (all when unset).
bool ShaderLogWanted(uint64_t hash);
bool GraphicsDebugDumpEnabled();
LogDirection GetPrintfDirection();
bool SpirvDebugPrintfEnabled();
bool GpuAssistedValidationEnabled();
bool ReadbackLinearImagesEnabled();
bool ReadbackTiledImagesEnabled();
bool ReadbackStorageImagesEnabled();
bool SyncRawImageBuffersEnabled();
uint32_t GetFlushDraws();
bool TessellationEnabled();
bool OcclusionQueriesEnabled();
bool RelaxedShaderFloatEnabled();
bool NativeRtEnabled();
bool LodStatsEnabled();

} // namespace Config

#endif /* KYTY_COMMON_EMULATOR_CONFIG_H_ */
