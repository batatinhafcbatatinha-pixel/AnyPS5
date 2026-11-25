#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/srwLock.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <map>
#include <mutex>
#include <span>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	// LDS storage of early (side queue) dispatches, which run beside the main queue's.
	[[nodiscard]] StreamBuffer& GetSideSharedMemoryBuffer();
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	// Another GPU write to a range all of whose bytes are still GPU-written (not downloaded, not
	// adopted from the CPU since) in one cache buffer: what ObtainBuffer(is_written) would set is
	// already set, so only the write tick and the color-metadata checks change. False otherwise.
	[[nodiscard]] bool RepeatGpuWrite(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// GPU thread: a command-processor write to guest memory, recorded in GPU order into the cache
	// buffer instead of written by the CPU, when the range lies on GPU-modified pages (a CPU write
	// there faults and waits for the GPU to finish the batch). Its bytes become GPU-modified, so
	// later CPU readers read them back once the GPU has reached the write. False, with nothing
	// done, where only the CPU write is known to be exact: the caller writes guest memory then.
	[[nodiscard]] bool WriteInGpuOrder(uint64_t vaddr, const void* data, uint64_t size);
	// GPU thread, after a command-processor fill or copy recorded in GPU order: the dwords it
	// leaves pending, when the CPU knows them, so walks read them without waiting (see
	// m_cp_known). KnownBytes: the bytes such a copy reads, when the CPU knows them now.
	void               NoteCpKnown(uint64_t vaddr, const void* data, uint64_t size);
	void               NoteCpFill(uint64_t vaddr, uint64_t size, uint32_t value);
	[[nodiscard]] bool KnownBytes(uint64_t vaddr, uint64_t size, void* data);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// Color metadata the GPU found free of the clear keys in `keys` (bits of code >> 5) stays so
	// until a GPU write or download.
	void               MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys);
	[[nodiscard]] bool IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const;
	void               ProcessFaultBuffer();
	// Uploads the CPU-dirty bytes of the queued ranges `mapped` covers behind one barrier pair.
	void SynchronizeBda(const RangeSet& mapped);
	// The unwatched pages whose bytes changed since the last scan (thread-safe); a submission
	// queues them for the BDA upload when its processing starts.
	[[nodiscard]] RangeSet ScanUnwatchedPages();
	void                   QueueUnwatchedPages(const RangeSet& pages);
	// A resumed submission rescans at its next BDA upload, not when it is retried: a retry that
	// blocks again reads nothing.
	void RequestUnwatchedRescan() { m_bda_rescan_pending.store(true, std::memory_order_release); }
	// Queue after the range's state change becomes visible, or a drain could consume it first.
	// is_write: a CPU write (the host view keeps the cache-buffer copy mapped and refreshes it).
	void               QueueBdaSync(uint64_t vaddr, uint64_t size, bool is_write = false);
	// Guest memory holds the newest bytes of the range, such as a downloaded image's: GPU writes
	// cached there are older and are dropped, so the next use reads guest memory.
	void               AdoptGuestMemory(uint64_t vaddr, uint64_t size);
	void               RunGarbageCollector();
	// Host view (GPU-SRT): BDA reads and writes reach guest memory as the CPU sees it, through
	// imported chunks of the guest backing, except on pages whose newest bytes a GPU write left
	// in a cache buffer (GPU-owned), which the page table sends to that buffer.
	[[nodiscard]] bool HostViewEnabled() const noexcept { return !m_host_chunks_failed && m_host_view; }
	// Called before the GPU thread submits a batch. Downloads the batch's GPU writes to pages
	// the CPU has read back before, so a later read waits only for this batch instead of
	// flushing and draining the queue.
	void               RecordEagerDownloads();
	[[nodiscard]] bool HasEagerWrites() const { return !m_eager_writes.Empty(); }
	// SRT-walker load on the GPU thread from a GPU-modified page: the guest word, when no GPU
	// write covers it, so the walk needs no read-back.
	[[nodiscard]] bool TryCurrentWalkerLoad(uint64_t vaddr, uint32_t& value);
	// GPU thread: whether the bytes hold a GPU write guest memory does not have yet, after
	// publishing the finished eager downloads that cover them. Never waits for the GPU.
	[[nodiscard]] bool HasPendingGpuWrite(uint64_t vaddr, uint64_t size);
	// GPU thread: reads a small table of guest memory the GPU may have written (see
	// LibKernel::Memory::ReadGpuTable).
	[[nodiscard]] bool ReadTable(uint64_t vaddr, uint64_t size, void* data);

	// Early compute (CommandScheduler::BeginSide): a compute program whose output the GPU thread
	// keeps waiting for while the main queue drains earlier work is a learned early producer.
	// Reads through BDA are not bound ranges: early compute writes nothing while one is queued.
	void SetCurrentProgram(uint64_t hash, bool reads_bda) noexcept;
	[[nodiscard]] bool IsEarlyProducer(uint64_t hash) const { return m_early_producers.contains(hash); }
	struct EarlyRange {
		uint64_t vaddr;
		uint64_t size;
		bool     written;
	};
	// 0 when a dispatch binding these ranges may run before the queued main work: no queued batch
	// writes them, none reads what it writes or what it uploads, and its written pages need no new
	// BDA entries. Else the reason.
	[[nodiscard]] int CheckEarly(std::span<const EarlyRange> ranges);
	// While recording the side buffer: downloads of its writes to read-back pages, published once
	// the side value is reached instead of with the main batch.
	void RecordSideDownloads(std::span<const EarlyRange> ranges, uint64_t side_value);

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const { return !id || m_slot_buffers[id].is_deleted; }

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	// Copies [previous upload's end, end) go to `buffer`.
	struct BufferUpload {
		Buffer* buffer;
		size_t  end;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	// Records the downloads of read-back writes (RecordEagerDownloads) in one batch.
	void RecordEagerBatch(const RangeSet& writes);
	// generations_out: see MemoryTracker::ForEachUploadRange.
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer,
	                                     uint64_t* generations_out = nullptr);
	void UploadCopies(std::span<const BufferUpload> uploads, std::span<vk::BufferCopy> copies,
	                  uint64_t total_size);
	// Guest bytes for an upload of tracker-reported pages: CopySparseMemory, but exact hot pages
	// (hotPages.h) through their shadow, which the copy refreshes.
	[[nodiscard]] bool CopyGuestForUpload(uint64_t address, uint8_t* destination, uint64_t size);
	// SynchronizeBda's changed exact hot pages (GPU thread).
	std::vector<uint64_t> m_hot_scan_scratch;
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void               ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size);
	// Publishes the eager downloads overlapping the range, waiting for them unless only the
	// finished ones are wanted, and makes their pages readable again.
	void               CompleteEagerDownloads(uint64_t vaddr, uint64_t size, bool finished_only,
	                                          bool side_only = false);
	// A GPU-thread wait for read-back bytes: learns their producer as an early producer.
	void               NoteProducerWait(uint64_t vaddr, uint64_t size, double ms);
	[[nodiscard]] bool HasEagerDownloads(uint64_t vaddr, uint64_t size) const;

	// A publication that runs once: on the priority thread in recording order, or earlier on a
	// thread that waits for it (after every older publication of overlapping bytes).
	struct OncePublish {
		std::mutex                   mutex;
		bool                         done = false;
		Common::UniqueFunction<void> run;
		void                         Run() {
			std::lock_guard lock(mutex);
			if (!done) {
				run();
				done = true;
			}
		}
	};
	struct EagerDownload {
		uint64_t begin;
		uint64_t end;
		uint64_t tick;
		uint64_t side = 0; // Recorded on the side queue: wait for this side value, then publish.
		std::shared_ptr<OncePublish> publish;
	};
	struct ProducerWindow {
		std::chrono::steady_clock::time_point start;
		uint32_t                              waits = 0;
		double                                ms    = 0;
	};

	// ObtainBuffer's last buffer-path result per range (GPU thread). A call repeated while
	// nothing it reads changed does nothing but set the buffer's ticks: the range is CPU-clean
	// (and GPU-dirty when written) already, the buffer still owns it, and the bookkeeping sets
	// it adds to hold it already. Tracker state is checked by region generations; the cache's
	// own sets and buffers by m_obtain_state, which grows whenever one could make a repeated
	// call do more: a buffer created or deleted, GPU-written bytes leaving
	// m_gpu_modified_ranges, color metadata checks added, read-back pages added. Read-back
	// bookkeeping of written ranges is redone on every call (NoteReadbackWrite).
	struct ObtainMemo {
		uint64_t vaddr       = 0;
		uint64_t size        = 0;
		uint64_t generations = 0;
		uint64_t state       = 0; // 0: empty
		BufferId id {};
		bool     written  = false;
		bool     readback = false; // a written range on read-back pages (NoteReadbackWrite)
	};
	static constexpr size_t OBTAIN_MEMO_SLOTS = 2048;
	[[nodiscard]] ObtainMemo& ObtainMemoSlot(uint64_t vaddr, uint64_t size, bool written);
	// ObtainBuffer of a written range: queues the eager download of its bytes on read-back pages
	// and names the current program as their writer. Whether there were any.
	bool NoteReadbackWrite(uint64_t vaddr, uint64_t size);
	void InvalidateObtainMemos() noexcept { m_obtain_state++; }
	std::unique_ptr<std::array<ObtainMemo, OBTAIN_MEMO_SLOTS>> m_obtain_memo;
	uint64_t                                          m_obtain_state = 1;

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	// Pages the CPU read while they held GPU writes, the GPU writes to them since the last
	// submit, and the downloads of those writes whose bytes guest memory does not hold yet.
	RangeSet                                          m_readback_pages;
	RangeSet                                          m_eager_writes;
	std::vector<EagerDownload>                        m_eager_downloads;
	RangeSet                                          m_eager_pending;
	bool                                              m_recording_eager = false;
	// CompleteEagerDownloads' pages to unmark, kept for its next call (GPU thread).
	std::vector<GuestRange>                           m_unmark_scratch;
	// Early compute: the program binding buffers now, the writers of read-back pages, learning.
	uint64_t                                          m_current_program = 0;
	// GPU thread: dwords a command-processor write recorded in GPU order (WriteInGpuOrder) set
	// last, by address; a GPU write over them (ObtainBuffer written) or a CPU write adopted
	// as the newest bytes (AdoptGuestMemory) drops them.
	static constexpr size_t      CpKnownLimit = 65536;
	std::map<uint64_t, uint32_t> m_cp_known;
	std::mutex                   m_cp_known_mutex; // AdoptGuestMemory may run on other threads.
	std::atomic<bool>            m_cp_known_any {false};
	void                         ForgetCpKnownWords(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool           CpKnownWords(uint64_t vaddr, uint64_t size, void* data);
	uint64_t                                          m_bda_reader_tick = 0;
	std::map<uint64_t, std::pair<uint64_t, uint64_t>> m_readback_writers; // begin -> {end, program}
	std::unordered_map<uint64_t, ProducerWindow>      m_producer_windows;
	std::unordered_set<uint64_t>                      m_early_producers;
	Common::UniqueFunction<void>*                     m_capture_publish = nullptr;
	// One per TileManager::ColorClearSlots key slot.
	std::array<RangeSet, 8>                           m_checked_color_metadata;
	MemoryTracker                                     m_memory_tracker;
	Common::SrwMutex                                  m_bda_sync_mutex;
	RangeSet                                          m_bda_sync;
	// Unwatched pages: no fault queues their writes, so each guest submission queues them again
	// (guarded by m_bda_sync_mutex).
	RangeSet                                          m_bda_unwatched;
	std::atomic<bool>                                 m_bda_unwatched_any {false};
	// Hash of the bytes each unwatched page had when it was last queued (guarded by
	// m_bda_scan_mutex); cleared when a buffer is created, whose first bytes come from another path.
	Common::SrwMutex                                  m_bda_scan_mutex;
	std::unordered_map<uint64_t, uint64_t>            m_bda_unwatched_hash;
	std::atomic<uint64_t>                             m_bda_hash_generation {0};
	uint64_t                                          m_bda_hash_seen = 0;
	// Small ranges streamed in the current guest submission: key (address, size, alignment) -> offset.
	// AnyPS5: open addressing, emptied by a new generation (was an unordered_map: a heap node per
	// streamed range and a full clear per submission).
	struct StreamMemoEntry {
		uint64_t key        = 0;
		uint64_t offset     = 0;
		uint64_t generation = 0;
	};
	static constexpr size_t                           STREAM_MEMO_SIZE = 16384; // power of two
	std::vector<StreamMemoEntry>                      m_stream_memo = std::vector<StreamMemoEntry>(STREAM_MEMO_SIZE);
	uint64_t                                          m_stream_memo_generation = 1;
	size_t                                            m_stream_memo_count      = 0;
	uint64_t                                          m_stream_memo_epoch = 0;
	uint64_t                                          m_stream_memo_last  = 0; // Wraps() the memo is for.
	// Host view: imported 32 MiB chunks of the guest backing by backing offset, the pages a GPU
	// write owns, and the pages whose BDA entries need rewriting (guarded by m_bda_sync_mutex).
	struct HostChunk {
		vk::Buffer        buffer  = nullptr;
		vk::DeviceMemory  memory  = nullptr;
		vk::DeviceAddress address = 0;
	};
	static constexpr uint64_t HostChunkBits = 25;
	std::unordered_map<uint64_t, HostChunk>           m_host_chunks;
	bool                                              m_host_view          = false;
	bool                                              m_host_chunks_failed = false;
	RangeSet                                          m_gpu_owned;
	RangeSet                                          m_bda_entry_pending;
	void              InitHostView();
	void              DestroyHostView();
	vk::DeviceAddress HostAddress(uint64_t vaddr);
	// Rewrites the page-table entries of whole pages in [begin, end); GPU thread. Changed entries
	// are queued and written together by FlushBdaEntries, before the next DMA draw reads them.
	void              WriteBdaEntries(uint64_t begin, uint64_t end);
	void              FlushBdaEntries();
	std::unordered_map<uint64_t, vk::DeviceAddress>   m_pt_shadow; // Page index -> written entry.
	std::vector<std::pair<uint64_t, vk::DeviceAddress>> m_pt_queue;
	// Set after m_bda_sync grows, so draws with nothing queued skip SynchronizeBda.
	std::atomic<bool>                                 m_bda_sync_pending {false};
	std::atomic<bool>                                 m_bda_rescan_pending {false};
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	std::unique_ptr<StreamBuffer>                     m_side_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
