#ifndef APS5_KYTY_SHIM_LIBS_AGC_H
#define APS5_KYTY_SHIM_LIBS_AGC_H

#include <cstdint>

namespace Libs::Graphics {

// Command buffer dumps (KytyPS5 --graphics-debug-dump) are not part of AnyPS5.
inline void GraphicsDbgDumpDcb(const char* /*type*/, uint32_t /*num_dw*/, const uint32_t* /*cmd_buffer*/) {}

} // namespace Libs::Graphics

namespace Libs::Graphics::Gen5 {

// User-config register writes that carry driver data instead of a register value. AnyPS5's
// libSceAgc writes none, so nothing is skipped.
inline bool AgcIsInternalDataPacket(uint32_t /*cmd_id*/, const uint32_t* /*payload*/) {
	return false;
}

} // namespace Libs::Graphics::Gen5

#endif
