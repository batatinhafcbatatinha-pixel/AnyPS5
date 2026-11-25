#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_CUTSWITCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_CUTSWITCH_H_

#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

// Changes ported from KytyPS5 (8ebe9a09, b386cf25, 0a1b544d) that wait for a game run on AnyPS5.
// AnyPS5: every letter is OFF by default; APS5_KYTY_X_CUT_ON lists the letters to turn on
// (APS5_KYTY_X_CUT_ON=all turns on every one but g, which is what KytyPS5 runs by default).
// APS5_KYTY_X_CUT lists letters to keep off and wins over APS5_KYTY_X_CUT_ON (KytyPS5's KYTY_X_CUT:
// there every cut but g is on unless listed). Read once per cut; call sites keep the result in a
// static.
//   a: ObtainBuffer reuses the result of an identical earlier call while nothing it depends on changed
//   b: recorder records sized to their command, not to the largest one
//   c: recorder's per-command queue check answered from the last open command buffer
//   d: eager read-back downloads share one copy list per batch
//   e: pages the CPU rewrites every frame stay unwatched, compared with a shadow of their upload
//   f: recorder thread reads the ring's write position once per burst, not per record
//   g: page protections that make pages stricter (BDA re-watch, GPU-modified marks) and the
//      BDA upload bytes behind them are applied once, before the batch is submitted. OFF unless
//      APS5_KYTY_X_CUT_ON lists it: other watchers of a pending page (texture cache, image heap) skip
//      the protection they need while it waits, so guest reads can miss a write-back.
//   h: CP-known overlay skips written ranges outside the addresses of its words
//   i: ObtainBuffer memos survive read-back downloads of other bytes and repeated marks
//   j: ObtainBuffer memo hits skip read-back bookkeeping that would change nothing
// AnyPS5 letters for ported changes that KytyPS5 runs without a switch or with one of its own:
//   k: command-processor writes over pending GPU bytes recorded in GPU order, and walks reading
//      the values those writes, DMA fills and copies of known bytes left (b386cf25)
//   p: SRT-walker condition loads on pending GPU words read as unknown, and thread-count indirect
//      dispatches whose counts a pending GPU write holds derived on the GPU (8ebe9a09)
//   s: scalar SRT reads on pending GPU words loaded by the shader through device addresses
//      (b386cf25; APS5_KYTY_GPU_SCALAR=0 also keeps it off)
//   w: single-wave wave64 compute as one lane per invocation (0a1b544d; APS5_KYTY_X_RT=w also
//      keeps it off)
[[nodiscard]] inline bool CutEnabled(char letter) noexcept {
	const char* disabled = std::getenv("APS5_KYTY_X_CUT");
	if (disabled != nullptr && std::strchr(disabled, letter) != nullptr) {
		return false;
	}
	const char* enabled = std::getenv("APS5_KYTY_X_CUT_ON");
	if (enabled == nullptr) {
		return false;
	}
	return std::strchr(enabled, letter) != nullptr || (letter != 'g' && std::strcmp(enabled, "all") == 0);
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_CUTSWITCH_H_
