#ifndef APS5_KYTY_SHIM_KERNEL_EVENTQUEUE_H
#define APS5_KYTY_SHIM_KERNEL_EVENTQUEUE_H

// The GPU backend does not own event queues: AnyPS5's libSceAgcDriver registers graphics events
// (AGC/Eq) and the backend delivers end-of-pipe interrupts through it (see glue/Bridge.hpp).
// These declarations only keep KytyPS5's registration helpers compiling; they are not called.

#include "common/abi.h"

#include <cstdint>
#include <deque>
#include <memory>

namespace Libs::LibKernel::EventQueue {

constexpr int16_t KERNEL_EVFILT_GRAPHICS = -14;

struct KernelEqueueEvent;

using KernelEqueue = int64_t;
constexpr KernelEqueue KERNEL_EQUEUE_INVALID = 0;

using trigger_func_t = void (*)(KernelEqueueEvent* event, void* trigger_data);
using reset_func_t   = void (*)(KernelEqueueEvent* event);
using delete_func_t  = void (*)(KernelEqueue eq, KernelEqueueEvent* event);

struct KernelEvent {
	uintptr_t ident  = 0;
	int16_t   filter = 0;
	uint16_t  flags  = 0;
	uint32_t  fflags = 0;
	intptr_t  data   = 0;
	void*     udata  = nullptr;
};

struct KernelFilter {
	void*                 data = nullptr;
	std::shared_ptr<void> owner;
	trigger_func_t        trigger_func      = nullptr;
	reset_func_t          reset_func        = nullptr;
	delete_func_t         delete_event_func = nullptr;
};

struct KernelEqueueEvent {
	bool                    triggered   = false;
	uint64_t                deadline_ns = 0;
	uint64_t                interval_ns = 0;
	KernelEvent             event;
	KernelFilter            filter;
	std::deque<KernelEvent> pending_events;
};

int KernelAddEvent(KernelEqueue eq, const KernelEqueueEvent& event);
int KernelTriggerEvent(KernelEqueue eq, uintptr_t ident, int16_t filter, void* trigger_data);
int KernelDeleteEvent(KernelEqueue eq, uintptr_t ident, int16_t filter);

} // namespace Libs::LibKernel::EventQueue

#endif
