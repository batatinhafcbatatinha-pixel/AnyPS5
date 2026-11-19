#include "graphics/host_gpu/renderer/gpuDiag.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/deviceLossReport.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	m_graphics.device.destroySemaphore(m_semaphore, nullptr);
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	if (result != vk::Result::eSuccess) {
		ReportDeviceLoss(m_graphics, "semaphore counter", static_cast<int>(result));
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
	// AnyPS5 diagnostic: blocking time per caller (APS5_KYTY_MEMSTATS report).
	struct Timed {
		void*                                 at;
		std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		~Timed() {
			if (Common::DebugCounters::Enabled()) {
				Common::DebugCounters::NoteWaitSite(
				    at, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				                                  std::chrono::steady_clock::now() - start)
				                                  .count()));
			}
		}
	} timed {__builtin_return_address(0)};
	// AnyPS5: waits in 5 s slices and reports a wait that does not end (diagnostic for hangs).
	auto result = m_graphics.device.waitSemaphores(&wait_info, 5'000'000'000ull);
	for (uint32_t slices = 1; result == vk::Result::eTimeout; ++slices) {
		uint64_t counter = 0;
		(void)m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
		if (slices == 2) GpuDiag::Report(m_graphics, "a GPU wait of 10 s");
		if (slices <= 3) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "[gpu-wait] semaphore {} waits for tick {} for {} s: counter {}, next tick {}\n",
			    static_cast<const void*>(this), tick, slices * 5, counter, CurrentTick()));
		}
		result = m_graphics.device.waitSemaphores(&wait_info, 5'000'000'000ull);
	}
	if (result != vk::Result::eSuccess) {
		ReportDeviceLoss(m_graphics, "semaphore wait", static_cast<int>(result));
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
}

} // namespace Libs::Graphics
