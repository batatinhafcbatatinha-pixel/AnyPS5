#ifndef APS5_KYTY_SHIM_LIBS_AGC_H
#define APS5_KYTY_SHIM_LIBS_AGC_H

#include <cstdint>

namespace Libs::Graphics {

// Command buffer dumps (KytyPS5 --graphics-debug-dump) are not part of AnyPS5.
inline void GraphicsDbgDumpDcb(const char* /*type*/, uint32_t /*num_dw*/, const uint32_t* /*cmd_buffer*/) {}

} // namespace Libs::Graphics

namespace Libs::Graphics::Gen5 {

// User-config register writes (SQ_THREAD_TRACE_USERDATA_2) that carry AnyPS5 driver data instead
// of a register value: libSceAgc's wait-on-address and multi-draw brackets and libSceAgcDriver's
// user-data markers. Adapted from KytyPS5 src/libs/agc.cpp AgcIsInternalDataPacket (3a7d8030).
inline bool AgcIsInternalDataPacket(uint32_t cmd_id, const uint32_t* payload) {
	constexpr uint32_t UserData2 = 0x342u;
	if (payload == nullptr || ((cmd_id >> 8u) & 0xffu) != 0x79u || payload[0] != UserData2) {
		return false;
	}
	const auto size_dw = ((cmd_id >> 16u) & 0x3fffu) + 2u;
	const auto tag     = payload[1];
	if (size_dw == 4u) {
		const auto begin = tag & 0xffff0000u;
		return begin == 0xc8010000u || begin == 0xc8020000u;
	}
	if (size_dw != 3u) {
		return false;
	}
	switch (tag & 0xff000000u) {
		case 0xc0000000u:
		case 0xc1000000u:
		case 0xc2000000u: return true;
		default: break;
	}
	return tag == 0xc8000000u || tag == 0xc6000000u || tag == 0xc6000005u || tag == 0xc6000006u ||
	       tag == 0xc6000008u;
}

} // namespace Libs::Graphics::Gen5

#endif
