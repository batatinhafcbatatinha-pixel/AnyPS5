// AnyPS5: Vulkan command recording on a second thread.
//
// The GPU thread spends a large part of every draw inside the Vulkan driver recording commands.
// Command buffers begun on the GPU thread are "deferred": their vkCmd* calls (Vulkan-Hpp calls
// them through the dynamic dispatcher, whose function pointers are replaced here) copy their
// arguments, including the arrays and structures they point to, into a ring; a recorder thread
// replays them in order into the real command buffer. vkEndCommandBuffer, command pool
// operations and the destruction of objects a pending command may use wait until the recorder
// has replayed everything queued so far. Commands with extension structures (pNext) are not
// copied: they also wait, then run on the calling thread.
// APS5_KYTY_RECORDER=0 records everything on the calling thread as before.

#include "glue/Recorder.hpp"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/vulkanCommon.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <intrin.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <thread>
#include <type_traits>

namespace AgcDriver::VramBackend::Recorder {
namespace {

// ------------------------------------------------------------------------------------- the ring

// Single producer (the GPU thread), single consumer (the recorder thread).
class Ring final {
public:
	static constexpr size_t Bytes    = size_t {32} << 20;
	static constexpr size_t SlotSize = 192; // the replay closure

	struct Header {
		void (*run)(void* closure);
		uint32_t size;   // whole record, header included; 0: wrap to the start
		uint32_t unused; // keeps the closure 16-byte aligned
	};

	Ring(): m_data(static_cast<uint8_t*>(_aligned_malloc(Bytes, 64))) {
		m_wake = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		m_thread = std::thread([this] { Run(); });
	}

	// Reserves a record with `extra` bytes after the closure; returns its start.
	uint8_t* Reserve(size_t extra) {
		const size_t size = (sizeof(Header) + SlotSize + extra + 15) & ~size_t {15};
		EXIT_IF(size > Bytes / 4);
		for (;;) {
			auto       write = m_write_local;
			auto       read  = m_read_cached;
			if (Bytes - (write - read) < size * 2 + 16) {
				read = m_read_cached = m_read.load(std::memory_order_acquire);
			}
			const auto used  = write - read;
			const auto at    = write % Bytes;
			// A record never wraps: the tail is skipped with a wrap marker.
			const size_t need = (at + size > Bytes) ? (Bytes - at) + size : size;
			if (Bytes - used >= need + 16) {
				if (at + size > Bytes) {
					auto* wrap = reinterpret_cast<Header*>(m_data + at);
					wrap->run  = nullptr;
					wrap->size = 0;
					m_write_local += Bytes - at;
				}
				m_pending_size = size;
				return m_data + (m_write_local % Bytes);
			}
			Kick();
			SpinOrYield();
		}
	}

	template <typename F>
	void Publish(uint8_t* record, F&& closure) {
		using Closure = std::decay_t<F>;
		static_assert(sizeof(Closure) <= SlotSize && alignof(Closure) <= 16);
		auto* header = reinterpret_cast<Header*>(record);
		new (record + sizeof(Header)) Closure(std::forward<F>(closure));
		header->run = [](void* at) {
			auto* stored = static_cast<Closure*>(at);
			(*stored)();
			stored->~Closure();
		};
		header->size = static_cast<uint32_t>(m_pending_size);
		m_write_local += m_pending_size;
		m_write.store(m_write_local, std::memory_order_release);
		if (m_sleeping.load(std::memory_order_seq_cst)) {
			Kick();
		}
	}

	// Waits until every published record ran.
	void Drain() {
		const auto target = m_write.load(std::memory_order_acquire);
		if (m_done.load(std::memory_order_acquire) >= target) {
			return;
		}
		Kick();
		while (m_done.load(std::memory_order_acquire) < target) {
			SpinOrYield();
		}
	}

	[[nodiscard]] bool OnRecorderThread() const noexcept { return GetCurrentThreadId() == m_thread_id.load(); }

private:
	void Kick() { SetEvent(m_wake); }

	static void SpinOrYield() {
		for (int i = 0; i < 64; ++i) {
			_mm_pause();
		}
		SwitchToThread();
	}

	void Run() {
		m_thread_id.store(GetCurrentThreadId());
		SetThreadDescription(GetCurrentThread(), L"Kyty_Recorder");
		uint64_t read = 0;
		for (;;) {
			auto write = m_write.load(std::memory_order_acquire);
			if (read == write) {
				// Spin a little (the next command usually follows within microseconds), then sleep.
				bool got = false;
				for (int i = 0; i < 4000 && !got; ++i) {
					_mm_pause();
					got = m_write.load(std::memory_order_acquire) != read;
				}
				if (!got) {
					m_sleeping.store(true, std::memory_order_seq_cst);
					if (m_write.load(std::memory_order_seq_cst) == read) {
						WaitForSingleObject(m_wake, 2);
					}
					m_sleeping.store(false, std::memory_order_relaxed);
				}
				continue;
			}
			while (read != write) {
				auto* header = reinterpret_cast<Header*>(m_data + (read % Bytes));
				if (header->size == 0) {
					read += Bytes - (read % Bytes);
					continue;
				}
				header->run(reinterpret_cast<uint8_t*>(header) + sizeof(Header));
				read += header->size;
				if ((++m_ran & 31) == 0) {
					m_read.store(read, std::memory_order_release);
					m_done.store(read, std::memory_order_release);
				}
				write = m_write.load(std::memory_order_acquire);
			}
			m_read.store(read, std::memory_order_release);
			m_done.store(read, std::memory_order_release);
		}
	}

	uint8_t*              m_data;
	HANDLE                m_wake = nullptr;
	std::atomic<uint64_t> m_write {0};
	std::atomic<uint64_t> m_read {0};
	std::atomic<uint64_t> m_done {0};
	std::atomic<bool>     m_sleeping {false};
	std::atomic<DWORD>    m_thread_id {0};
	uint64_t              m_ran = 0; // consumer only
	uint64_t              m_write_local  = 0; // producer only
	uint64_t              m_read_cached  = 0; // producer only
	size_t                m_pending_size = 0; // producer only
	std::thread           m_thread;
};

Ring* g_ring = nullptr;

// Open deferred command buffers (Begin on the GPU thread .. End).
constexpr size_t                                MaxOpen = 64;
std::array<std::atomic<VkCommandBuffer>, MaxOpen> g_open {};

std::atomic<size_t> g_open_high {0}; // slots in use are below this

bool IsOpen(VkCommandBuffer buffer) noexcept {
	const auto high = g_open_high.load(std::memory_order_acquire);
	for (size_t i = 0; i < high; ++i) {
		if (g_open[i].load(std::memory_order_acquire) == buffer) {
			return true;
		}
	}
	return false;
}
void Open(VkCommandBuffer buffer) {
	for (size_t i = 0; i < MaxOpen; ++i) {
		VkCommandBuffer expected = VK_NULL_HANDLE;
		if (g_open[i].compare_exchange_strong(expected, buffer)) {
			auto high = g_open_high.load();
			while (high < i + 1 && !g_open_high.compare_exchange_weak(high, i + 1)) {
			}
			return;
		}
	}
	EXIT("Recorder: too many open command buffers\n");
}
void Close(VkCommandBuffer buffer) {
	for (auto& slot: g_open) {
		VkCommandBuffer expected = buffer;
		if (slot.compare_exchange_strong(expected, VK_NULL_HANDLE)) {
			return;
		}
	}
}

// How a call on `buffer` runs: queued, or here (after the queue drained when the buffer is a
// deferred one used from another thread).
bool Queue(VkCommandBuffer buffer) {
	if (g_ring == nullptr || !IsOpen(buffer)) {
		return false;
	}
	if (Libs::Graphics::GuestGpu::IsGpuThread()) {
		return true;
	}
	g_ring->Drain();
	return false;
}

void DrainAll() {
	if (g_ring != nullptr && Libs::Graphics::GuestGpu::IsGpuThread()) {
		g_ring->Drain();
	} else if (g_ring != nullptr) {
		// Another thread destroying or resetting: the GPU thread's queue may still use the object.
		g_ring->Drain();
	}
}

// Copies of the arrays a command points to, placed after the closure in its record.
class Extra final {
public:
	explicit Extra(uint8_t* at): m_at(at) {}
	template <typename T>
	T* Copy(const T* from, size_t count) {
		if (from == nullptr || count == 0) {
			return nullptr;
		}
		m_at       = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(m_at) + alignof(T) - 1) &
		                                       ~(uintptr_t {alignof(T)} - 1));
		auto* copy = reinterpret_cast<T*>(m_at);
		std::memcpy(copy, from, sizeof(T) * count);
		m_at += sizeof(T) * count;
		return copy;
	}

private:
	uint8_t* m_at;
};
template <typename T>
constexpr size_t Room(size_t count) {
	return count == 0 ? 0 : sizeof(T) * count + alignof(T);
}

// Queues `run` (a closure over copies) with `extra` bytes of copies filled by `fill(Extra&)`.
template <typename Fill, typename Make>
void Push(size_t extra, Fill&& fill, Make&& make) {
	auto* record = g_ring->Reserve(extra);
	Extra copies(record + sizeof(Ring::Header) + Ring::SlotSize);
	auto  state = fill(copies);
	g_ring->Publish(record, make(state));
}
template <typename F>
void PushSimple(F&& run) {
	auto* record = g_ring->Reserve(0);
	g_ring->Publish(record, std::forward<F>(run));
}

// ------------------------------------------------------------------------- real entry points

struct Real {
#define APS5_REAL(name) PFN_##name name = nullptr;
	APS5_REAL(vkBeginCommandBuffer)
	APS5_REAL(vkEndCommandBuffer)
	APS5_REAL(vkResetCommandBuffer)
	APS5_REAL(vkAllocateCommandBuffers)
	APS5_REAL(vkFreeCommandBuffers)
	APS5_REAL(vkResetCommandPool)
	APS5_REAL(vkDestroyCommandPool)
	APS5_REAL(vkDestroyBuffer)
	APS5_REAL(vkDestroyImage)
	APS5_REAL(vkDestroyImageView)
	APS5_REAL(vkDestroyBufferView)
	APS5_REAL(vkDestroySampler)
	APS5_REAL(vkDestroyPipeline)
	APS5_REAL(vkDestroyPipelineLayout)
	APS5_REAL(vkDestroyDescriptorSetLayout)
	APS5_REAL(vkDestroyQueryPool)
	APS5_REAL(vkFreeMemory)
	APS5_REAL(vkGetDeviceProcAddr)
	APS5_REAL(vkCmdBeginConditionalRenderingEXT)
	APS5_REAL(vkCmdEndConditionalRenderingEXT)
	APS5_REAL(vkCmdBeginQuery)
	APS5_REAL(vkCmdEndQuery)
	APS5_REAL(vkCmdBeginRendering)
	APS5_REAL(vkCmdEndRendering)
	APS5_REAL(vkCmdBindDescriptorSets)
	APS5_REAL(vkCmdBindIndexBuffer)
	APS5_REAL(vkCmdBindPipeline)
	APS5_REAL(vkCmdBindVertexBuffers2)
	APS5_REAL(vkCmdBlitImage)
	APS5_REAL(vkCmdBuildAccelerationStructuresKHR)
	APS5_REAL(vkCmdClearAttachments)
	APS5_REAL(vkCmdClearColorImage)
	APS5_REAL(vkCmdClearDepthStencilImage)
	APS5_REAL(vkCmdCopyBuffer)
	APS5_REAL(vkCmdCopyBufferToImage)
	APS5_REAL(vkCmdCopyImage)
	APS5_REAL(vkCmdCopyImageToBuffer)
	APS5_REAL(vkCmdDispatch)
	APS5_REAL(vkCmdDispatchBase)
	APS5_REAL(vkCmdDispatchIndirect)
	APS5_REAL(vkCmdDraw)
	APS5_REAL(vkCmdDrawIndexed)
	APS5_REAL(vkCmdDrawIndexedIndirect)
	APS5_REAL(vkCmdDrawIndirect)
	APS5_REAL(vkCmdDrawMeshTasksEXT)
	APS5_REAL(vkCmdFillBuffer)
	APS5_REAL(vkCmdPipelineBarrier)
	APS5_REAL(vkCmdPipelineBarrier2)
	APS5_REAL(vkCmdPushConstants)
	APS5_REAL(vkCmdPushDescriptorSetKHR)
	APS5_REAL(vkCmdResolveImage)
	APS5_REAL(vkCmdSetAttachmentFeedbackLoopEnableEXT)
	APS5_REAL(vkCmdSetBlendConstants)
	APS5_REAL(vkCmdSetCullMode)
	APS5_REAL(vkCmdSetDepthBias)
	APS5_REAL(vkCmdSetDepthBiasEnable)
	APS5_REAL(vkCmdSetDepthBounds)
	APS5_REAL(vkCmdSetDepthBoundsTestEnable)
	APS5_REAL(vkCmdSetDepthCompareOp)
	APS5_REAL(vkCmdSetDepthTestEnable)
	APS5_REAL(vkCmdSetDepthWriteEnable)
	APS5_REAL(vkCmdSetFrontFace)
	APS5_REAL(vkCmdSetLineWidth)
	APS5_REAL(vkCmdSetScissor)
	APS5_REAL(vkCmdSetScissorWithCount)
	APS5_REAL(vkCmdSetStencilCompareMask)
	APS5_REAL(vkCmdSetStencilOp)
	APS5_REAL(vkCmdSetStencilReference)
	APS5_REAL(vkCmdSetStencilTestEnable)
	APS5_REAL(vkCmdSetStencilWriteMask)
	APS5_REAL(vkCmdSetViewport)
	APS5_REAL(vkCmdSetViewportWithCount)
	APS5_REAL(vkCmdUpdateBuffer)
#undef APS5_REAL
};
Real g_real;

// ----------------------------------------------------------------------------- command buffers

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer buffer, const VkCommandBufferBeginInfo* info) {
	if (g_ring == nullptr || !Libs::Graphics::GuestGpu::IsGpuThread() || info->pNext != nullptr ||
	    info->pInheritanceInfo != nullptr) {
		return g_real.vkBeginCommandBuffer(buffer, info);
	}
	Open(buffer);
	const auto flags = info->flags;
	PushSimple([buffer, flags] {
		VkCommandBufferBeginInfo begin {};
		begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin.flags = flags;
		const auto result = g_real.vkBeginCommandBuffer(buffer, &begin);
		EXIT_IF(result != VK_SUCCESS);
	});
	return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(VkCommandBuffer buffer) {
	if (g_ring != nullptr && IsOpen(buffer)) {
		g_ring->Drain();
		Close(buffer);
	}
	return g_real.vkEndCommandBuffer(buffer);
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(VkCommandBuffer buffer, VkCommandBufferResetFlags flags) {
	DrainAll();
	Close(buffer);
	return g_real.vkResetCommandBuffer(buffer, flags);
}
VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* info,
                                                      VkCommandBuffer* buffers) {
	DrainAll();
	return g_real.vkAllocateCommandBuffers(device, info, buffers);
}
VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(VkDevice device, VkCommandPool pool, uint32_t count,
                                              const VkCommandBuffer* buffers) {
	DrainAll();
	for (uint32_t i = 0; i < count; ++i) {
		Close(buffers[i]);
	}
	g_real.vkFreeCommandBuffers(device, pool, count, buffers);
}
VKAPI_ATTR VkResult VKAPI_CALL ResetCommandPool(VkDevice device, VkCommandPool pool, VkCommandPoolResetFlags flags) {
	DrainAll();
	return g_real.vkResetCommandPool(device, pool, flags);
}
VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(VkDevice device, VkCommandPool pool, const VkAllocationCallbacks* a) {
	DrainAll();
	g_real.vkDestroyCommandPool(device, pool, a);
}

#define APS5_DESTROY(name, Type)                                                                \
	VKAPI_ATTR void VKAPI_CALL name(VkDevice device, Type object, const VkAllocationCallbacks* a) { \
		DrainAll();                                                                             \
		g_real.vk##name(device, object, a);                                                     \
	}
APS5_DESTROY(DestroyBuffer, VkBuffer)
APS5_DESTROY(DestroyImage, VkImage)
APS5_DESTROY(DestroyImageView, VkImageView)
APS5_DESTROY(DestroyBufferView, VkBufferView)
APS5_DESTROY(DestroySampler, VkSampler)
APS5_DESTROY(DestroyPipeline, VkPipeline)
APS5_DESTROY(DestroyPipelineLayout, VkPipelineLayout)
APS5_DESTROY(DestroyDescriptorSetLayout, VkDescriptorSetLayout)
APS5_DESTROY(DestroyQueryPool, VkQueryPool)
APS5_DESTROY(FreeMemory, VkDeviceMemory)
#undef APS5_DESTROY

// ------------------------------------------------------------------------ commands with scalars

#define APS5_SCALAR(name, params, args, captures)                       \
	VKAPI_ATTR void VKAPI_CALL name params {                             \
		if (!Queue(buffer)) {                                            \
			g_real.vk##name args;                                        \
			return;                                                      \
		}                                                                \
		PushSimple([captures] { g_real.vk##name args; });                \
	}

#define APS5_P(...) (__VA_ARGS__)
#define APS5_C(...) __VA_ARGS__

APS5_SCALAR(CmdEndConditionalRenderingEXT, APS5_P(VkCommandBuffer buffer), APS5_P(buffer), APS5_C(buffer))
APS5_SCALAR(CmdBeginQuery, APS5_P(VkCommandBuffer buffer, VkQueryPool pool, uint32_t query, VkQueryControlFlags flags),
            APS5_P(buffer, pool, query, flags), APS5_C(buffer, pool, query, flags))
APS5_SCALAR(CmdEndQuery, APS5_P(VkCommandBuffer buffer, VkQueryPool pool, uint32_t query), APS5_P(buffer, pool, query),
            APS5_C(buffer, pool, query))
APS5_SCALAR(CmdEndRendering, APS5_P(VkCommandBuffer buffer), APS5_P(buffer), APS5_C(buffer))
APS5_SCALAR(CmdBindIndexBuffer, APS5_P(VkCommandBuffer buffer, VkBuffer index, VkDeviceSize offset, VkIndexType type),
            APS5_P(buffer, index, offset, type), APS5_C(buffer, index, offset, type))
APS5_SCALAR(CmdBindPipeline, APS5_P(VkCommandBuffer buffer, VkPipelineBindPoint point, VkPipeline pipeline),
            APS5_P(buffer, point, pipeline), APS5_C(buffer, point, pipeline))
APS5_SCALAR(CmdDispatch, APS5_P(VkCommandBuffer buffer, uint32_t x, uint32_t y, uint32_t z), APS5_P(buffer, x, y, z),
            APS5_C(buffer, x, y, z))
APS5_SCALAR(CmdDispatchBase,
            APS5_P(VkCommandBuffer buffer, uint32_t bx, uint32_t by, uint32_t bz, uint32_t x, uint32_t y, uint32_t z),
            APS5_P(buffer, bx, by, bz, x, y, z), APS5_C(buffer, bx, by, bz, x, y, z))
APS5_SCALAR(CmdDispatchIndirect, APS5_P(VkCommandBuffer buffer, VkBuffer args, VkDeviceSize offset),
            APS5_P(buffer, args, offset), APS5_C(buffer, args, offset))
APS5_SCALAR(CmdDraw, APS5_P(VkCommandBuffer buffer, uint32_t vertices, uint32_t instances, uint32_t first_vertex, uint32_t first_instance),
            APS5_P(buffer, vertices, instances, first_vertex, first_instance),
            APS5_C(buffer, vertices, instances, first_vertex, first_instance))
APS5_SCALAR(CmdDrawIndexed,
            APS5_P(VkCommandBuffer buffer, uint32_t indices, uint32_t instances, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance),
            APS5_P(buffer, indices, instances, first_index, vertex_offset, first_instance),
            APS5_C(buffer, indices, instances, first_index, vertex_offset, first_instance))
APS5_SCALAR(CmdDrawIndexedIndirect,
            APS5_P(VkCommandBuffer buffer, VkBuffer args, VkDeviceSize offset, uint32_t count, uint32_t stride),
            APS5_P(buffer, args, offset, count, stride), APS5_C(buffer, args, offset, count, stride))
APS5_SCALAR(CmdDrawIndirect,
            APS5_P(VkCommandBuffer buffer, VkBuffer args, VkDeviceSize offset, uint32_t count, uint32_t stride),
            APS5_P(buffer, args, offset, count, stride), APS5_C(buffer, args, offset, count, stride))
APS5_SCALAR(CmdDrawMeshTasksEXT, APS5_P(VkCommandBuffer buffer, uint32_t x, uint32_t y, uint32_t z), APS5_P(buffer, x, y, z),
            APS5_C(buffer, x, y, z))
APS5_SCALAR(CmdFillBuffer, APS5_P(VkCommandBuffer buffer, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size, uint32_t data),
            APS5_P(buffer, dst, offset, size, data), APS5_C(buffer, dst, offset, size, data))
APS5_SCALAR(CmdSetAttachmentFeedbackLoopEnableEXT, APS5_P(VkCommandBuffer buffer, VkImageAspectFlags mask),
            APS5_P(buffer, mask), APS5_C(buffer, mask))
APS5_SCALAR(CmdSetCullMode, APS5_P(VkCommandBuffer buffer, VkCullModeFlags mode), APS5_P(buffer, mode), APS5_C(buffer, mode))
APS5_SCALAR(CmdSetDepthBias, APS5_P(VkCommandBuffer buffer, float constant, float clamp, float slope),
            APS5_P(buffer, constant, clamp, slope), APS5_C(buffer, constant, clamp, slope))
APS5_SCALAR(CmdSetDepthBiasEnable, APS5_P(VkCommandBuffer buffer, VkBool32 enable), APS5_P(buffer, enable), APS5_C(buffer, enable))
APS5_SCALAR(CmdSetDepthBounds, APS5_P(VkCommandBuffer buffer, float low, float high), APS5_P(buffer, low, high),
            APS5_C(buffer, low, high))
APS5_SCALAR(CmdSetDepthBoundsTestEnable, APS5_P(VkCommandBuffer buffer, VkBool32 enable), APS5_P(buffer, enable),
            APS5_C(buffer, enable))
APS5_SCALAR(CmdSetDepthCompareOp, APS5_P(VkCommandBuffer buffer, VkCompareOp op), APS5_P(buffer, op), APS5_C(buffer, op))
APS5_SCALAR(CmdSetDepthTestEnable, APS5_P(VkCommandBuffer buffer, VkBool32 enable), APS5_P(buffer, enable), APS5_C(buffer, enable))
APS5_SCALAR(CmdSetDepthWriteEnable, APS5_P(VkCommandBuffer buffer, VkBool32 enable), APS5_P(buffer, enable), APS5_C(buffer, enable))
APS5_SCALAR(CmdSetFrontFace, APS5_P(VkCommandBuffer buffer, VkFrontFace face), APS5_P(buffer, face), APS5_C(buffer, face))
APS5_SCALAR(CmdSetLineWidth, APS5_P(VkCommandBuffer buffer, float width), APS5_P(buffer, width), APS5_C(buffer, width))
APS5_SCALAR(CmdSetStencilCompareMask, APS5_P(VkCommandBuffer buffer, VkStencilFaceFlags faces, uint32_t mask),
            APS5_P(buffer, faces, mask), APS5_C(buffer, faces, mask))
APS5_SCALAR(CmdSetStencilOp,
            APS5_P(VkCommandBuffer buffer, VkStencilFaceFlags faces, VkStencilOp fail, VkStencilOp pass, VkStencilOp depth_fail, VkCompareOp op),
            APS5_P(buffer, faces, fail, pass, depth_fail, op), APS5_C(buffer, faces, fail, pass, depth_fail, op))
APS5_SCALAR(CmdSetStencilReference, APS5_P(VkCommandBuffer buffer, VkStencilFaceFlags faces, uint32_t reference),
            APS5_P(buffer, faces, reference), APS5_C(buffer, faces, reference))
APS5_SCALAR(CmdSetStencilTestEnable, APS5_P(VkCommandBuffer buffer, VkBool32 enable), APS5_P(buffer, enable),
            APS5_C(buffer, enable))
APS5_SCALAR(CmdSetStencilWriteMask, APS5_P(VkCommandBuffer buffer, VkStencilFaceFlags faces, uint32_t mask),
            APS5_P(buffer, faces, mask), APS5_C(buffer, faces, mask))

// ------------------------------------------------------------------------- commands with arrays

VKAPI_ATTR void VKAPI_CALL CmdSetBlendConstants(VkCommandBuffer buffer, const float constants[4]) {
	if (!Queue(buffer)) {
		g_real.vkCmdSetBlendConstants(buffer, constants);
		return;
	}
	std::array<float, 4> copy {constants[0], constants[1], constants[2], constants[3]};
	PushSimple([buffer, copy] { g_real.vkCmdSetBlendConstants(buffer, copy.data()); });
}

VKAPI_ATTR void VKAPI_CALL CmdSetViewport(VkCommandBuffer buffer, uint32_t first, uint32_t count, const VkViewport* viewports) {
	if (!Queue(buffer)) {
		g_real.vkCmdSetViewport(buffer, first, count, viewports);
		return;
	}
	Push(Room<VkViewport>(count), [&](Extra& extra) { return extra.Copy(viewports, count); },
	     [&](const VkViewport* copy) { return [buffer, first, count, copy] { g_real.vkCmdSetViewport(buffer, first, count, copy); }; });
}
VKAPI_ATTR void VKAPI_CALL CmdSetScissor(VkCommandBuffer buffer, uint32_t first, uint32_t count, const VkRect2D* rects) {
	if (!Queue(buffer)) {
		g_real.vkCmdSetScissor(buffer, first, count, rects);
		return;
	}
	Push(Room<VkRect2D>(count), [&](Extra& extra) { return extra.Copy(rects, count); },
	     [&](const VkRect2D* copy) { return [buffer, first, count, copy] { g_real.vkCmdSetScissor(buffer, first, count, copy); }; });
}
VKAPI_ATTR void VKAPI_CALL CmdSetViewportWithCount(VkCommandBuffer buffer, uint32_t count, const VkViewport* viewports) {
	if (!Queue(buffer)) {
		g_real.vkCmdSetViewportWithCount(buffer, count, viewports);
		return;
	}
	Push(Room<VkViewport>(count), [&](Extra& extra) { return extra.Copy(viewports, count); },
	     [&](const VkViewport* copy) { return [buffer, count, copy] { g_real.vkCmdSetViewportWithCount(buffer, count, copy); }; });
}
VKAPI_ATTR void VKAPI_CALL CmdSetScissorWithCount(VkCommandBuffer buffer, uint32_t count, const VkRect2D* rects) {
	if (!Queue(buffer)) {
		g_real.vkCmdSetScissorWithCount(buffer, count, rects);
		return;
	}
	Push(Room<VkRect2D>(count), [&](Extra& extra) { return extra.Copy(rects, count); },
	     [&](const VkRect2D* copy) { return [buffer, count, copy] { g_real.vkCmdSetScissorWithCount(buffer, count, copy); }; });
}

VKAPI_ATTR void VKAPI_CALL CmdPushConstants(VkCommandBuffer buffer, VkPipelineLayout layout, VkShaderStageFlags stages,
                                            uint32_t offset, uint32_t size, const void* values) {
	if (!Queue(buffer)) {
		g_real.vkCmdPushConstants(buffer, layout, stages, offset, size, values);
		return;
	}
	Push(Room<uint8_t>(size), [&](Extra& extra) { return extra.Copy(static_cast<const uint8_t*>(values), size); },
	     [&](const uint8_t* copy) {
		     return [buffer, layout, stages, offset, size, copy] {
			     g_real.vkCmdPushConstants(buffer, layout, stages, offset, size, copy);
		     };
	     });
}

VKAPI_ATTR void VKAPI_CALL CmdUpdateBuffer(VkCommandBuffer buffer, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size,
                                           const void* data) {
	if (!Queue(buffer)) {
		g_real.vkCmdUpdateBuffer(buffer, dst, offset, size, data);
		return;
	}
	Push(Room<uint8_t>(size), [&](Extra& extra) { return extra.Copy(static_cast<const uint8_t*>(data), size); },
	     [&](const uint8_t* copy) {
		     return [buffer, dst, offset, size, copy] { g_real.vkCmdUpdateBuffer(buffer, dst, offset, size, copy); };
	     });
}

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(VkCommandBuffer buffer, VkPipelineBindPoint point, VkPipelineLayout layout,
                                                 uint32_t first, uint32_t count, const VkDescriptorSet* sets,
                                                 uint32_t dynamic_count, const uint32_t* dynamic) {
	if (!Queue(buffer)) {
		g_real.vkCmdBindDescriptorSets(buffer, point, layout, first, count, sets, dynamic_count, dynamic);
		return;
	}
	struct State {
		const VkDescriptorSet* sets;
		const uint32_t*        dynamic;
	};
	Push(Room<VkDescriptorSet>(count) + Room<uint32_t>(dynamic_count),
	     [&](Extra& extra) { return State {extra.Copy(sets, count), extra.Copy(dynamic, dynamic_count)}; },
	     [&](const State& s) {
		     return [buffer, point, layout, first, count, dynamic_count, s] {
			     g_real.vkCmdBindDescriptorSets(buffer, point, layout, first, count, s.sets, dynamic_count, s.dynamic);
		     };
	     });
}

VKAPI_ATTR void VKAPI_CALL CmdBindVertexBuffers2(VkCommandBuffer buffer, uint32_t first, uint32_t count, const VkBuffer* buffers,
                                                 const VkDeviceSize* offsets, const VkDeviceSize* sizes,
                                                 const VkDeviceSize* strides) {
	if (!Queue(buffer)) {
		g_real.vkCmdBindVertexBuffers2(buffer, first, count, buffers, offsets, sizes, strides);
		return;
	}
	struct State {
		const VkBuffer*     buffers;
		const VkDeviceSize* offsets;
		const VkDeviceSize* sizes;
		const VkDeviceSize* strides;
	};
	Push(Room<VkBuffer>(count) + 3 * Room<VkDeviceSize>(count),
	     [&](Extra& extra) {
		     return State {extra.Copy(buffers, count), extra.Copy(offsets, count), extra.Copy(sizes, count),
		                   extra.Copy(strides, count)};
	     },
	     [&](const State& s) {
		     return [buffer, first, count, s] {
			     g_real.vkCmdBindVertexBuffers2(buffer, first, count, s.buffers, s.offsets, s.sizes, s.strides);
		     };
	     });
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer buffer, VkBuffer src, VkBuffer dst, uint32_t count,
                                         const VkBufferCopy* regions) {
	if (!Queue(buffer)) {
		g_real.vkCmdCopyBuffer(buffer, src, dst, count, regions);
		return;
	}
	Push(Room<VkBufferCopy>(count), [&](Extra& extra) { return extra.Copy(regions, count); },
	     [&](const VkBufferCopy* copy) {
		     return [buffer, src, dst, count, copy] { g_real.vkCmdCopyBuffer(buffer, src, dst, count, copy); };
	     });
}
VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage(VkCommandBuffer buffer, VkBuffer src, VkImage dst, VkImageLayout layout,
                                                uint32_t count, const VkBufferImageCopy* regions) {
	if (!Queue(buffer)) {
		g_real.vkCmdCopyBufferToImage(buffer, src, dst, layout, count, regions);
		return;
	}
	Push(Room<VkBufferImageCopy>(count), [&](Extra& extra) { return extra.Copy(regions, count); },
	     [&](const VkBufferImageCopy* copy) {
		     return [buffer, src, dst, layout, count, copy] {
			     g_real.vkCmdCopyBufferToImage(buffer, src, dst, layout, count, copy);
		     };
	     });
}
VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer(VkCommandBuffer buffer, VkImage src, VkImageLayout layout, VkBuffer dst,
                                                uint32_t count, const VkBufferImageCopy* regions) {
	if (!Queue(buffer)) {
		g_real.vkCmdCopyImageToBuffer(buffer, src, layout, dst, count, regions);
		return;
	}
	Push(Room<VkBufferImageCopy>(count), [&](Extra& extra) { return extra.Copy(regions, count); },
	     [&](const VkBufferImageCopy* copy) {
		     return [buffer, src, layout, dst, count, copy] {
			     g_real.vkCmdCopyImageToBuffer(buffer, src, layout, dst, count, copy);
		     };
	     });
}
VKAPI_ATTR void VKAPI_CALL CmdCopyImage(VkCommandBuffer buffer, VkImage src, VkImageLayout src_layout, VkImage dst,
                                        VkImageLayout dst_layout, uint32_t count, const VkImageCopy* regions) {
	if (!Queue(buffer)) {
		g_real.vkCmdCopyImage(buffer, src, src_layout, dst, dst_layout, count, regions);
		return;
	}
	Push(Room<VkImageCopy>(count), [&](Extra& extra) { return extra.Copy(regions, count); },
	     [&](const VkImageCopy* copy) {
		     return [buffer, src, src_layout, dst, dst_layout, count, copy] {
			     g_real.vkCmdCopyImage(buffer, src, src_layout, dst, dst_layout, count, copy);
		     };
	     });
}
VKAPI_ATTR void VKAPI_CALL CmdBlitImage(VkCommandBuffer buffer, VkImage src, VkImageLayout src_layout, VkImage dst,
                                        VkImageLayout dst_layout, uint32_t count, const VkImageBlit* regions, VkFilter filter) {
	if (!Queue(buffer)) {
		g_real.vkCmdBlitImage(buffer, src, src_layout, dst, dst_layout, count, regions, filter);
		return;
	}
	Push(Room<VkImageBlit>(count), [&](Extra& extra) { return extra.Copy(regions, count); },
	     [&](const VkImageBlit* copy) {
		     return [buffer, src, src_layout, dst, dst_layout, count, copy, filter] {
			     g_real.vkCmdBlitImage(buffer, src, src_layout, dst, dst_layout, count, copy, filter);
		     };
	     });
}
VKAPI_ATTR void VKAPI_CALL CmdResolveImage(VkCommandBuffer buffer, VkImage src, VkImageLayout src_layout, VkImage dst,
                                           VkImageLayout dst_layout, uint32_t count, const VkImageResolve* regions) {
	if (!Queue(buffer)) {
		g_real.vkCmdResolveImage(buffer, src, src_layout, dst, dst_layout, count, regions);
		return;
	}
	Push(Room<VkImageResolve>(count), [&](Extra& extra) { return extra.Copy(regions, count); },
	     [&](const VkImageResolve* copy) {
		     return [buffer, src, src_layout, dst, dst_layout, count, copy] {
			     g_real.vkCmdResolveImage(buffer, src, src_layout, dst, dst_layout, count, copy);
		     };
	     });
}
VKAPI_ATTR void VKAPI_CALL CmdClearColorImage(VkCommandBuffer buffer, VkImage image, VkImageLayout layout,
                                              const VkClearColorValue* color, uint32_t count,
                                              const VkImageSubresourceRange* ranges) {
	if (!Queue(buffer)) {
		g_real.vkCmdClearColorImage(buffer, image, layout, color, count, ranges);
		return;
	}
	const VkClearColorValue value = *color;
	Push(Room<VkImageSubresourceRange>(count), [&](Extra& extra) { return extra.Copy(ranges, count); },
	     [&](const VkImageSubresourceRange* copy) {
		     return [buffer, image, layout, value, count, copy] {
			     g_real.vkCmdClearColorImage(buffer, image, layout, &value, count, copy);
		     };
	     });
}
VKAPI_ATTR void VKAPI_CALL CmdClearDepthStencilImage(VkCommandBuffer buffer, VkImage image, VkImageLayout layout,
                                                     const VkClearDepthStencilValue* value, uint32_t count,
                                                     const VkImageSubresourceRange* ranges) {
	if (!Queue(buffer)) {
		g_real.vkCmdClearDepthStencilImage(buffer, image, layout, value, count, ranges);
		return;
	}
	const VkClearDepthStencilValue clear = *value;
	Push(Room<VkImageSubresourceRange>(count), [&](Extra& extra) { return extra.Copy(ranges, count); },
	     [&](const VkImageSubresourceRange* copy) {
		     return [buffer, image, layout, clear, count, copy] {
			     g_real.vkCmdClearDepthStencilImage(buffer, image, layout, &clear, count, copy);
		     };
	     });
}
VKAPI_ATTR void VKAPI_CALL CmdClearAttachments(VkCommandBuffer buffer, uint32_t count, const VkClearAttachment* attachments,
                                               uint32_t rect_count, const VkClearRect* rects) {
	if (!Queue(buffer)) {
		g_real.vkCmdClearAttachments(buffer, count, attachments, rect_count, rects);
		return;
	}
	struct State {
		const VkClearAttachment* attachments;
		const VkClearRect*       rects;
	};
	Push(Room<VkClearAttachment>(count) + Room<VkClearRect>(rect_count),
	     [&](Extra& extra) { return State {extra.Copy(attachments, count), extra.Copy(rects, rect_count)}; },
	     [&](const State& s) {
		     return [buffer, count, rect_count, s] {
			     g_real.vkCmdClearAttachments(buffer, count, s.attachments, rect_count, s.rects);
		     };
	     });
}

// ------------------------------------------------------------------- commands with structures

VKAPI_ATTR void VKAPI_CALL CmdBeginConditionalRenderingEXT(VkCommandBuffer buffer,
                                                           const VkConditionalRenderingBeginInfoEXT* info) {
	if (info->pNext != nullptr || !Queue(buffer)) {
		if (g_ring != nullptr && IsOpen(buffer)) {
			g_ring->Drain();
		}
		g_real.vkCmdBeginConditionalRenderingEXT(buffer, info);
		return;
	}
	const auto copy = *info;
	PushSimple([buffer, copy] { g_real.vkCmdBeginConditionalRenderingEXT(buffer, &copy); });
}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(VkCommandBuffer buffer, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                              VkDependencyFlags flags, uint32_t memory_count, const VkMemoryBarrier* memory,
                                              uint32_t buffer_count, const VkBufferMemoryBarrier* buffers,
                                              uint32_t image_count, const VkImageMemoryBarrier* images) {
	bool chained = false;
	for (uint32_t i = 0; i < memory_count; ++i) chained |= memory[i].pNext != nullptr;
	for (uint32_t i = 0; i < buffer_count; ++i) chained |= buffers[i].pNext != nullptr;
	for (uint32_t i = 0; i < image_count; ++i) chained |= images[i].pNext != nullptr;
	if (chained || !Queue(buffer)) {
		if (g_ring != nullptr && IsOpen(buffer)) {
			g_ring->Drain();
		}
		g_real.vkCmdPipelineBarrier(buffer, src, dst, flags, memory_count, memory, buffer_count, buffers, image_count, images);
		return;
	}
	struct State {
		const VkMemoryBarrier*       memory;
		const VkBufferMemoryBarrier* buffers;
		const VkImageMemoryBarrier*  images;
	};
	Push(Room<VkMemoryBarrier>(memory_count) + Room<VkBufferMemoryBarrier>(buffer_count) +
	         Room<VkImageMemoryBarrier>(image_count),
	     [&](Extra& extra) {
		     return State {extra.Copy(memory, memory_count), extra.Copy(buffers, buffer_count),
		                   extra.Copy(images, image_count)};
	     },
	     [&](const State& s) {
		     return [buffer, src, dst, flags, memory_count, buffer_count, image_count, s] {
			     g_real.vkCmdPipelineBarrier(buffer, src, dst, flags, memory_count, s.memory, buffer_count, s.buffers,
			                                 image_count, s.images);
		     };
	     });
}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier2(VkCommandBuffer buffer, const VkDependencyInfo* info) {
	bool chained = info->pNext != nullptr;
	for (uint32_t i = 0; i < info->memoryBarrierCount; ++i) chained |= info->pMemoryBarriers[i].pNext != nullptr;
	for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; ++i) chained |= info->pBufferMemoryBarriers[i].pNext != nullptr;
	for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) chained |= info->pImageMemoryBarriers[i].pNext != nullptr;
	if (chained || !Queue(buffer)) {
		if (g_ring != nullptr && IsOpen(buffer)) {
			g_ring->Drain();
		}
		g_real.vkCmdPipelineBarrier2(buffer, info);
		return;
	}
	Push(sizeof(VkDependencyInfo) + 16 + Room<VkMemoryBarrier2>(info->memoryBarrierCount) +
	         Room<VkBufferMemoryBarrier2>(info->bufferMemoryBarrierCount) +
	         Room<VkImageMemoryBarrier2>(info->imageMemoryBarrierCount),
	     [&](Extra& extra) {
		     auto* copy                  = extra.Copy(info, 1);
		     copy->pMemoryBarriers       = extra.Copy(info->pMemoryBarriers, info->memoryBarrierCount);
		     copy->pBufferMemoryBarriers = extra.Copy(info->pBufferMemoryBarriers, info->bufferMemoryBarrierCount);
		     copy->pImageMemoryBarriers  = extra.Copy(info->pImageMemoryBarriers, info->imageMemoryBarrierCount);
		     return copy;
	     },
	     [&](const VkDependencyInfo* copy) { return [buffer, copy] { g_real.vkCmdPipelineBarrier2(buffer, copy); }; });
}

VKAPI_ATTR void VKAPI_CALL CmdBeginRendering(VkCommandBuffer buffer, const VkRenderingInfo* info) {
	bool chained = info->pNext != nullptr;
	for (uint32_t i = 0; i < info->colorAttachmentCount; ++i) chained |= info->pColorAttachments[i].pNext != nullptr;
	chained |= info->pDepthAttachment != nullptr && info->pDepthAttachment->pNext != nullptr;
	chained |= info->pStencilAttachment != nullptr && info->pStencilAttachment->pNext != nullptr;
	if (chained || !Queue(buffer)) {
		if (g_ring != nullptr && IsOpen(buffer)) {
			g_ring->Drain();
		}
		g_real.vkCmdBeginRendering(buffer, info);
		return;
	}
	Push(sizeof(VkRenderingInfo) + 16 + Room<VkRenderingAttachmentInfo>(info->colorAttachmentCount + 2),
	     [&](Extra& extra) {
		     auto* copy              = extra.Copy(info, 1);
		     copy->pColorAttachments = extra.Copy(info->pColorAttachments, info->colorAttachmentCount);
		     copy->pDepthAttachment  = extra.Copy(info->pDepthAttachment, info->pDepthAttachment != nullptr ? 1 : 0);
		     copy->pStencilAttachment = extra.Copy(info->pStencilAttachment, info->pStencilAttachment != nullptr ? 1 : 0);
		     return copy;
	     },
	     [&](const VkRenderingInfo* copy) { return [buffer, copy] { g_real.vkCmdBeginRendering(buffer, copy); }; });
}

VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSetKHR(VkCommandBuffer buffer, VkPipelineBindPoint point, VkPipelineLayout layout,
                                                   uint32_t set, uint32_t count, const VkWriteDescriptorSet* writes) {
	bool   chained = false;
	size_t extra   = Room<VkWriteDescriptorSet>(count);
	for (uint32_t i = 0; i < count; ++i) {
		chained |= writes[i].pNext != nullptr;
		extra += Room<VkDescriptorImageInfo>(writes[i].pImageInfo != nullptr ? writes[i].descriptorCount : 0) +
		         Room<VkDescriptorBufferInfo>(writes[i].pBufferInfo != nullptr ? writes[i].descriptorCount : 0) +
		         Room<VkBufferView>(writes[i].pTexelBufferView != nullptr ? writes[i].descriptorCount : 0);
	}
	if (chained || !Queue(buffer)) {
		if (g_ring != nullptr && IsOpen(buffer)) {
			g_ring->Drain();
		}
		g_real.vkCmdPushDescriptorSetKHR(buffer, point, layout, set, count, writes);
		return;
	}
	Push(extra,
	     [&](Extra& copies) {
		     auto* copy = copies.Copy(writes, count);
		     for (uint32_t i = 0; i < count; ++i) {
			     const auto n          = copy[i].descriptorCount;
			     copy[i].pImageInfo    = copies.Copy(writes[i].pImageInfo, writes[i].pImageInfo != nullptr ? n : 0);
			     copy[i].pBufferInfo   = copies.Copy(writes[i].pBufferInfo, writes[i].pBufferInfo != nullptr ? n : 0);
			     copy[i].pTexelBufferView =
			         copies.Copy(writes[i].pTexelBufferView, writes[i].pTexelBufferView != nullptr ? n : 0);
		     }
		     return copy;
	     },
	     [&](const VkWriteDescriptorSet* copy) {
		     return [buffer, point, layout, set, count, copy] {
			     g_real.vkCmdPushDescriptorSetKHR(buffer, point, layout, set, count, copy);
		     };
	     });
}

VKAPI_ATTR void VKAPI_CALL CmdBuildAccelerationStructuresKHR(VkCommandBuffer buffer, uint32_t count,
                                                             const VkAccelerationStructureBuildGeometryInfoKHR* infos,
                                                             const VkAccelerationStructureBuildRangeInfoKHR* const* ranges) {
	if (g_ring != nullptr && IsOpen(buffer)) {
		g_ring->Drain();
	}
	g_real.vkCmdBuildAccelerationStructuresKHR(buffer, count, infos, ranges);
}

// ----------------------------------------------------------------------------------- install

struct Entry {
	const char* name;
	PFN_vkVoidFunction stub;
};
#define APS5_ENTRY(name) {"vk" #name, reinterpret_cast<PFN_vkVoidFunction>(&name)}
const Entry kEntries[] = {
    APS5_ENTRY(BeginCommandBuffer), APS5_ENTRY(EndCommandBuffer), APS5_ENTRY(ResetCommandBuffer),
    APS5_ENTRY(AllocateCommandBuffers), APS5_ENTRY(FreeCommandBuffers), APS5_ENTRY(ResetCommandPool),
    APS5_ENTRY(DestroyCommandPool), APS5_ENTRY(DestroyBuffer), APS5_ENTRY(DestroyImage), APS5_ENTRY(DestroyImageView),
    APS5_ENTRY(DestroyBufferView), APS5_ENTRY(DestroySampler), APS5_ENTRY(DestroyPipeline),
    APS5_ENTRY(DestroyPipelineLayout), APS5_ENTRY(DestroyDescriptorSetLayout), APS5_ENTRY(DestroyQueryPool),
    APS5_ENTRY(FreeMemory),
};
#undef APS5_ENTRY

PFN_vkGetDeviceProcAddr g_get_device_proc = nullptr;

// VMA fetches its functions through this: it gets the draining destroy and free functions too.
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char* name) {
	for (const auto& entry: kEntries) {
		if (std::strcmp(entry.name, name) == 0) {
			return entry.stub;
		}
	}
	return g_get_device_proc(device, name);
}

} // namespace

void Install() {
	const char* setting = std::getenv("APS5_KYTY_RECORDER");
	if (setting != nullptr && std::strcmp(setting, "0") == 0) {
		Log::WriteToConsoleAndLog("[recorder] off (APS5_KYTY_RECORDER=0)\n");
		return;
	}
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
#define APS5_HOOK(name)                                         \
	g_real.vk##name = d.vk##name;                               \
	if (g_real.vk##name == nullptr) {                           \
		Log::WriteToConsoleAndLog("[recorder] vk" #name " missing: off\n"); \
		return;                                                 \
	}
#define APS5_HOOK_OPTIONAL(name) g_real.vk##name = d.vk##name;
	APS5_HOOK(BeginCommandBuffer)
	APS5_HOOK(EndCommandBuffer)
	APS5_HOOK(ResetCommandBuffer)
	APS5_HOOK(AllocateCommandBuffers)
	APS5_HOOK(FreeCommandBuffers)
	APS5_HOOK(ResetCommandPool)
	APS5_HOOK(DestroyCommandPool)
	APS5_HOOK(DestroyBuffer)
	APS5_HOOK(DestroyImage)
	APS5_HOOK(DestroyImageView)
	APS5_HOOK(DestroyBufferView)
	APS5_HOOK(DestroySampler)
	APS5_HOOK(DestroyPipeline)
	APS5_HOOK(DestroyPipelineLayout)
	APS5_HOOK(DestroyDescriptorSetLayout)
	APS5_HOOK(DestroyQueryPool)
	APS5_HOOK(FreeMemory)
	APS5_HOOK(GetDeviceProcAddr)
	APS5_HOOK_OPTIONAL(CmdBeginConditionalRenderingEXT)
	APS5_HOOK_OPTIONAL(CmdEndConditionalRenderingEXT)
	APS5_HOOK(CmdBeginQuery)
	APS5_HOOK(CmdEndQuery)
	APS5_HOOK(CmdBeginRendering)
	APS5_HOOK(CmdEndRendering)
	APS5_HOOK(CmdBindDescriptorSets)
	APS5_HOOK(CmdBindIndexBuffer)
	APS5_HOOK(CmdBindPipeline)
	APS5_HOOK(CmdBindVertexBuffers2)
	APS5_HOOK(CmdBlitImage)
	APS5_HOOK_OPTIONAL(CmdBuildAccelerationStructuresKHR)
	APS5_HOOK(CmdClearAttachments)
	APS5_HOOK(CmdClearColorImage)
	APS5_HOOK(CmdClearDepthStencilImage)
	APS5_HOOK(CmdCopyBuffer)
	APS5_HOOK(CmdCopyBufferToImage)
	APS5_HOOK(CmdCopyImage)
	APS5_HOOK(CmdCopyImageToBuffer)
	APS5_HOOK(CmdDispatch)
	APS5_HOOK(CmdDispatchBase)
	APS5_HOOK(CmdDispatchIndirect)
	APS5_HOOK(CmdDraw)
	APS5_HOOK(CmdDrawIndexed)
	APS5_HOOK(CmdDrawIndexedIndirect)
	APS5_HOOK(CmdDrawIndirect)
	APS5_HOOK_OPTIONAL(CmdDrawMeshTasksEXT)
	APS5_HOOK(CmdFillBuffer)
	APS5_HOOK(CmdPipelineBarrier)
	APS5_HOOK(CmdPipelineBarrier2)
	APS5_HOOK(CmdPushConstants)
	APS5_HOOK(CmdPushDescriptorSetKHR)
	APS5_HOOK(CmdResolveImage)
	APS5_HOOK_OPTIONAL(CmdSetAttachmentFeedbackLoopEnableEXT)
	APS5_HOOK(CmdSetBlendConstants)
	APS5_HOOK(CmdSetCullMode)
	APS5_HOOK(CmdSetDepthBias)
	APS5_HOOK(CmdSetDepthBiasEnable)
	APS5_HOOK(CmdSetDepthBounds)
	APS5_HOOK(CmdSetDepthBoundsTestEnable)
	APS5_HOOK(CmdSetDepthCompareOp)
	APS5_HOOK(CmdSetDepthTestEnable)
	APS5_HOOK(CmdSetDepthWriteEnable)
	APS5_HOOK(CmdSetFrontFace)
	APS5_HOOK(CmdSetLineWidth)
	APS5_HOOK(CmdSetScissor)
	APS5_HOOK(CmdSetScissorWithCount)
	APS5_HOOK(CmdSetStencilCompareMask)
	APS5_HOOK(CmdSetStencilOp)
	APS5_HOOK(CmdSetStencilReference)
	APS5_HOOK(CmdSetStencilTestEnable)
	APS5_HOOK(CmdSetStencilWriteMask)
	APS5_HOOK(CmdSetViewport)
	APS5_HOOK(CmdSetViewportWithCount)
	APS5_HOOK(CmdUpdateBuffer)
#undef APS5_HOOK
#undef APS5_HOOK_OPTIONAL

	g_ring            = new Ring();
	g_get_device_proc = g_real.vkGetDeviceProcAddr;

#define APS5_SET(name) d.vk##name = &name;
#define APS5_SET_OPTIONAL(name) \
	if (g_real.vk##name != nullptr) d.vk##name = &name;
	APS5_SET(BeginCommandBuffer)
	APS5_SET(EndCommandBuffer)
	APS5_SET(ResetCommandBuffer)
	APS5_SET(AllocateCommandBuffers)
	APS5_SET(FreeCommandBuffers)
	APS5_SET(ResetCommandPool)
	APS5_SET(DestroyCommandPool)
	APS5_SET(DestroyBuffer)
	APS5_SET(DestroyImage)
	APS5_SET(DestroyImageView)
	APS5_SET(DestroyBufferView)
	APS5_SET(DestroySampler)
	APS5_SET(DestroyPipeline)
	APS5_SET(DestroyPipelineLayout)
	APS5_SET(DestroyDescriptorSetLayout)
	APS5_SET(DestroyQueryPool)
	APS5_SET(FreeMemory)
	APS5_SET(GetDeviceProcAddr)
	APS5_SET_OPTIONAL(CmdBeginConditionalRenderingEXT)
	APS5_SET_OPTIONAL(CmdEndConditionalRenderingEXT)
	APS5_SET(CmdBeginQuery)
	APS5_SET(CmdEndQuery)
	APS5_SET(CmdBeginRendering)
	APS5_SET(CmdEndRendering)
	APS5_SET(CmdBindDescriptorSets)
	APS5_SET(CmdBindIndexBuffer)
	APS5_SET(CmdBindPipeline)
	APS5_SET(CmdBindVertexBuffers2)
	APS5_SET(CmdBlitImage)
	APS5_SET_OPTIONAL(CmdBuildAccelerationStructuresKHR)
	APS5_SET(CmdClearAttachments)
	APS5_SET(CmdClearColorImage)
	APS5_SET(CmdClearDepthStencilImage)
	APS5_SET(CmdCopyBuffer)
	APS5_SET(CmdCopyBufferToImage)
	APS5_SET(CmdCopyImage)
	APS5_SET(CmdCopyImageToBuffer)
	APS5_SET(CmdDispatch)
	APS5_SET(CmdDispatchBase)
	APS5_SET(CmdDispatchIndirect)
	APS5_SET(CmdDraw)
	APS5_SET(CmdDrawIndexed)
	APS5_SET(CmdDrawIndexedIndirect)
	APS5_SET(CmdDrawIndirect)
	APS5_SET_OPTIONAL(CmdDrawMeshTasksEXT)
	APS5_SET(CmdFillBuffer)
	APS5_SET(CmdPipelineBarrier)
	APS5_SET(CmdPipelineBarrier2)
	APS5_SET(CmdPushConstants)
	APS5_SET(CmdPushDescriptorSetKHR)
	APS5_SET(CmdResolveImage)
	APS5_SET_OPTIONAL(CmdSetAttachmentFeedbackLoopEnableEXT)
	APS5_SET(CmdSetBlendConstants)
	APS5_SET(CmdSetCullMode)
	APS5_SET(CmdSetDepthBias)
	APS5_SET(CmdSetDepthBiasEnable)
	APS5_SET(CmdSetDepthBounds)
	APS5_SET(CmdSetDepthBoundsTestEnable)
	APS5_SET(CmdSetDepthCompareOp)
	APS5_SET(CmdSetDepthTestEnable)
	APS5_SET(CmdSetDepthWriteEnable)
	APS5_SET(CmdSetFrontFace)
	APS5_SET(CmdSetLineWidth)
	APS5_SET(CmdSetScissor)
	APS5_SET(CmdSetScissorWithCount)
	APS5_SET(CmdSetStencilCompareMask)
	APS5_SET(CmdSetStencilOp)
	APS5_SET(CmdSetStencilReference)
	APS5_SET(CmdSetStencilTestEnable)
	APS5_SET(CmdSetStencilWriteMask)
	APS5_SET(CmdSetViewport)
	APS5_SET(CmdSetViewportWithCount)
	APS5_SET(CmdUpdateBuffer)
#undef APS5_SET
#undef APS5_SET_OPTIONAL
	// The KHR/EXT aliases Vulkan-Hpp may call.
	d.vkCmdBeginRenderingKHR          = d.vkCmdBeginRendering;
	d.vkCmdEndRenderingKHR            = d.vkCmdEndRendering;
	d.vkCmdPipelineBarrier2KHR        = d.vkCmdPipelineBarrier2;
	d.vkCmdBindVertexBuffers2EXT      = d.vkCmdBindVertexBuffers2;
	d.vkCmdSetCullModeEXT             = d.vkCmdSetCullMode;
	d.vkCmdSetFrontFaceEXT            = d.vkCmdSetFrontFace;
	d.vkCmdSetDepthTestEnableEXT      = d.vkCmdSetDepthTestEnable;
	d.vkCmdSetDepthWriteEnableEXT     = d.vkCmdSetDepthWriteEnable;
	d.vkCmdSetDepthCompareOpEXT       = d.vkCmdSetDepthCompareOp;
	d.vkCmdSetDepthBoundsTestEnableEXT = d.vkCmdSetDepthBoundsTestEnable;
	d.vkCmdSetStencilTestEnableEXT    = d.vkCmdSetStencilTestEnable;
	d.vkCmdSetStencilOpEXT            = d.vkCmdSetStencilOp;
	d.vkCmdSetDepthBiasEnableEXT      = d.vkCmdSetDepthBiasEnable;
	d.vkCmdSetViewportWithCountEXT    = d.vkCmdSetViewportWithCount;
	d.vkCmdSetScissorWithCountEXT     = d.vkCmdSetScissorWithCount;
	d.vkCmdDispatchBaseKHR            = d.vkCmdDispatchBase;
	Log::WriteToConsoleAndLog("[recorder] Vulkan commands of the GPU thread are recorded on their own thread\n");
}

} // namespace AgcDriver::VramBackend::Recorder
