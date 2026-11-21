#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           EndRendering() { m_command.EndRendering(); }
	void           Flush();
	void           Flush(SubmitInfo& submit);
	// Called before each dispatch the GPU thread records: submits the open batch once it holds a
	// few dispatches and the GPU has finished every batch before it, so the GPU does not sit idle
	// while the batch keeps growing.
	void           SubmitIfIdle();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Whether every priority operation of `tick` and earlier ticks has run, without waiting.
	[[nodiscard]] bool        PriorityOperationsDone(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	// range covers every guest access of the callback: an empty range declares a host-only
	// callback, an omitted one conservatively aliases all guest memory.
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation,
	                                                 std::optional<GuestRange> range = std::nullopt);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	// Queued or running priority callbacks that may access the range. A snapshot, not a barrier.
	[[nodiscard]] bool        HasPendingPriorityOperations(GuestRange range);
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

	// Early compute: one dispatch recorded into a command buffer of its own and submitted at once
	// to the second queue, so it runs before the work already queued on the main queue. The next
	// main submission waits for it, so everything recorded after it (and every resource retired
	// with the current tick) stays ordered after it.
	[[nodiscard]] bool     SideAvailable() const noexcept { return m_graphics.side_queue != nullptr; }
	// Side queue present and KYTY_EARLY_COMPUTE is not 0.
	[[nodiscard]] bool     EarlyComputeEnabled() const noexcept;
	[[nodiscard]] bool     InSide() const noexcept { return m_in_side; }
	// The side value the current side recording signals.
	[[nodiscard]] uint64_t SideValue() const noexcept { return m_side.CurrentTick(); }
	void                   BeginSide();
	uint64_t               SubmitSide();
	void                   WaitSide(uint64_t value) {
		SubmitQueuedUpTo(UINT64_MAX); // Side batches can wait for queued main batches.
		m_side.Wait(value);
	}
	[[nodiscard]] bool     IsSideFree(uint64_t value) {
		m_side.Refresh();
		return m_side.IsFree(value);
	}

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
		std::optional<GuestRange>    range; // Priority callbacks only, see DeferPriorityOperation.
	};

	void BeginNext();
	void PriorityOperationsThread(std::stop_token stop);
	// Queue submission of the GPU thread's batches runs on a worker, in tick order.
	struct QueuedSubmit {
		SubmitInfo        info;
		vk::CommandBuffer buffer = nullptr;
		uint64_t          tick   = 0;
	};
	void SubmitThread(std::stop_token stop);
	void SubmitToQueue(const SubmitInfo& submit, vk::CommandBuffer buffer, uint64_t tick);
	void DrainSubmits();
	void SubmitQueuedUpTo(uint64_t tick);
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority,
	                    std::optional<GuestRange> range = std::nullopt);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	MasterSemaphore              m_side;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandPool                  m_side_pool;
	vk::CommandBuffer            m_main_buffer = nullptr; // Saved while recording a side buffer.
	bool                         m_in_side     = false;
	uint32_t                     m_batch_dispatches = 0; // Dispatches recorded into the open batch.
	uint64_t                     m_side_wait   = 0; // Side value the next main submission waits for.
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations;
	std::deque<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	uint64_t                     m_priority_active_tick = 0; // 0 while none runs
	std::optional<GuestRange>    m_priority_active_range;    // of the running callback
	OperationState               m_operation_state      = OperationState::Open;
	std::deque<QueuedSubmit>     m_submits;
	std::mutex                   m_submit_mutex;
	std::condition_variable_any  m_submit_changed;
	bool                         m_submit_busy = false;
	bool                         m_async_submit = true; // KYTY_ASYNC_SUBMIT=0 submits on the calling thread
	std::jthread                 m_submit_thread; // Last: stops before the members it uses.
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
