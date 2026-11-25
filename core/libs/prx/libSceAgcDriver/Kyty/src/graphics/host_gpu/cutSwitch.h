#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_CUTSWITCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_CUTSWITCH_H_

#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

// GPU-thread CPU cuts that keep the same results can each be turned off without a rebuild, to
// find which one a behaviour change comes from: APS5_KYTY_X_CUT lists the letters of the cuts to turn
// off (APS5_KYTY_X_CUT=ac). All are on by default. Read once per cut; call sites keep the result in a
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
[[nodiscard]] inline bool CutEnabled(char letter) noexcept {
	if (letter == 'g') {
		const char* enabled = std::getenv("APS5_KYTY_X_CUT_ON");
		return enabled != nullptr && std::strchr(enabled, letter) != nullptr;
	}
	const char* disabled = std::getenv("APS5_KYTY_X_CUT");
	return disabled == nullptr || std::strchr(disabled, letter) == nullptr;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_CUTSWITCH_H_
