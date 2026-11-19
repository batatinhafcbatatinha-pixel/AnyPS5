#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAP_H_

#include "common/common.h"
#include "common/srwLock.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandBuffer;
class PageManager;
class RenderContext;
class RenderExecutor;
struct ImageHeapTestAccess;

// Guest image tables that shaders index by GPU-computed offsets, kept resident in one bindless
// sampled-image array. A table is keyed by its base and follows its descriptor's size. Each view
// of a table owns device slot words, one per table dword, that select an array element; one
// element serves a descriptor and profile in every view. CPU writes to a table fault in its
// snapshot, which each draw refreshes once; a view takes the changes when a draw first acquires
// it. Images behind elements report their changes through Image::heap_changes. A smaller table
// leaves at a suspend point if the frame did not acquire it, a larger one after
// LargeTableIdleFrames such frames. With GPU-SRT images, the heap also keeps the global
// descriptor map, whose image buckets select (and hold refs on) the same elements.
class ImageHeap final: public ShaderRecompiler::IR::ImageHeapSource {
public:
	ImageHeap(RenderContext& context, PageManager& page_manager);
	~ImageHeap();

	KYTY_CLASS_NO_COPY(ImageHeap);

	// GPU-SRT global descriptor map (ShaderRecompiler::IR::GlobalMap*): the entry of a T# or S#
	// key, resolved now, or UINT32_MAX when the map is off or the key finds no bucket.
	uint32_t GlobalEntry(
	    const std::array<uint32_t, ShaderRecompiler::IR::GlobalMapKeyWords>& key) override;
	// At submission: reads back the keys shaders of the batch reported missing and clears the
	// ring. Rebind resolves them once the batch has finished; Commit writes their buckets.
	void RecordGlobalReadback(CommandBuffer& buffer);

	// Starts a draw or dispatch, before its programs acquire views.
	void BeginDraw() { ++m_draw; }
	bool Acquire(const ShaderRecompiler::IR::ImageHeapRequest& request,
	             ShaderRecompiler::IR::ImageHeapView&          result) override;
	// Image discovery: resolves new elements and rediscovers changed images.
	void Rebind(RenderExecutor& executor);
	// Recording: uploads slots, moves resident images to their read layout and binds the array.
	void Commit(CommandBuffer& buffer, vk::PipelineBindPoint bind_point, vk::PipelineLayout layout);
	// CPU write fault, on any thread.
	void InvalidateMemory(uint64_t address, uint64_t size);
	void UnmapMemory(uint64_t address, uint64_t size);
	// At a suspend point.
	void RunGarbageCollector();

private:
	using DescriptorValue = ShaderRecompiler::IR::DescriptorValue;

	struct Key {
		DescriptorValue value;
		uint32_t        profile = 0; // Index into m_profiles.

		bool operator==(const Key&) const = default;
	};

	struct KeyHash {
		size_t operator()(const Key& key) const noexcept;
	};

	// A stale element's slots move to a new element of its descriptor, a buried one's to null.
	enum class State : uint8_t { Live, Stale, Buried };

	struct Element {
		Key      key;
		ImageId  image;
		uint64_t serial = 0; // Image::serial of image.
		uint32_t kind   = 0;
		uint32_t refs   = 0; // Slots of all views that select it.
		State    state  = State::Live;
	};

	struct Memory;

	struct View {
		Memory&                      memory;
		uint64_t                     step    = 0;
		uint64_t                     first   = 0;
		uint64_t                     count   = 0;
		uint64_t                     limit   = 0; // Probes a key domain allows, if fewer.
		uint32_t                     profile = 0;
		std::vector<uint32_t>        slots; // Plus a trailing null slot.
		std::vector<uint32_t>        kind_keys;
		std::vector<DescriptorValue> kinds; // One representative per kind_keys entry.
		std::unique_ptr<Buffer>      buffer;
		size_t                       dirty_begin   = 0; // Slots to upload.
		size_t                       dirty_end     = 0;
		size_t                       changed_begin = 0; // Table bytes changed since it was probed.
		size_t                       changed_end   = 0;
	};

	struct Memory {
		uint64_t                           base = 0;
		uint64_t                           size = 0;
		std::vector<uint32_t>              words;
		std::vector<uint8_t>               dirty_pages; // Unwatched, so unread since a write.
		std::unordered_map<uint32_t, uint64_t> dead;   // Buried words: their descriptor hashes.
		uint64_t                           draw = 0;    // Last draw that refreshed it.
		uint32_t                           idle = 0;    // Suspend points since a frame acquired it.
		std::vector<std::unique_ptr<View>> views;
	};

	[[nodiscard]] bool Refresh(Memory& memory);
	void               MarkDirty(Memory& memory, size_t begin, size_t end);
	template <bool track>
	void     Watch(const Memory& memory, size_t begin, size_t end);
	void     Unwatch(const Memory& memory);
	void     Drop(Memory& memory);
	void     UpdateProbes(View& view, uint64_t begin, uint64_t end);
	void     UpdateProbe(View& view, uint64_t probe);
	void     Assign(View& view, size_t slot, uint32_t entry);
	uint32_t Intern(View& view, uint32_t kind, const Key& key);
	uint32_t InternElement(uint32_t kind, const Key& key);
	void     Release(uint32_t entry);
	void     Retire();
	void     Mark(uint32_t index, State state);
	template <typename Fn>
	bool     ForEachChanged(Fn&& fn);
	void     Repoint();
	void     TakeChanges();
	ImageId  CurrentImage(const Image::HeapChange& change);
	uint32_t Allocate();
	void     Resize(Memory& memory, uint64_t base, uint64_t size);
	void     WriteDescriptors();

	using GlobalKey = std::array<uint32_t, ShaderRecompiler::IR::GlobalMapKeyWords>;
	// A resolved bucket of the global map. An image bucket holds a ref on its element, so the
	// element stays while the bucket selects it.
	struct GlobalBucket {
		GlobalKey key {};
		uint32_t  entry = 0; // Image: arm << ImageHeapKindShift | element; else sampler index.
		bool      known = false;
		bool      image = false;
	};
	struct GlobalReadback {
		uint64_t                tick       = 0;
		uint64_t                generation = 0;
		std::unique_ptr<Buffer> buffer;
	};
	void     DrainGlobalMisses();
	void     ResolveGlobalBucket(uint32_t bucket, const GlobalKey& key);
	void     ForgetGlobalBucket(uint32_t bucket);
	void     RepointGlobal();
	void     ResetGlobalMap();
	uint32_t GlobalProfile(uint32_t word);
	uint32_t GlobalSampler(const GlobalKey& key);
	void     ClearGlobalMap(CommandBuffer& buffer);
	void     UploadGlobalBuckets(CommandBuffer& buffer);

	RenderContext&                       m_context;
	PageManager&                         m_page_manager;
	Common::SrwMutex                     m_mutex; // Fault-visible Memory state and m_memories.
	std::vector<std::unique_ptr<Memory>> m_memories;
	std::vector<View*>                   m_active; // Acquired since the last commit.
	uint64_t                             m_draw         = 1;
	uint64_t                             m_suspend_draw = 0; // m_draw at the last suspend point.
	std::vector<ShaderRecompiler::IR::ImageResource>  m_profiles;
	std::vector<Element>                              m_elements;
	std::unordered_map<Key, uint32_t, KeyHash>        m_index;
	std::vector<uint32_t>                             m_free;
	std::vector<uint32_t>                             m_pending;  // Not yet resolved.
	std::vector<uint32_t>                             m_released; // Unselected after the upload.
	std::vector<uint32_t>                             m_held; // Another extent holds the address.
	std::vector<uint32_t>                             m_dcc;  // Metadata clears apply per draw.
	std::vector<std::pair<uint64_t, ImageId>>         m_away; // Serial, image: not read layout.
	uint64_t                                          m_held_serial = 0; // Image::serials.
	std::deque<std::pair<uint64_t, uint32_t>>         m_retired; // Tick, element.
	std::vector<std::pair<size_t, size_t>>            m_runs;
	std::vector<uint32_t>                             m_scratch;
	std::vector<Image::HeapChange>                    m_changes;
	std::vector<std::pair<uint32_t, vk::DescriptorImageInfo>> m_writes;
	vk::DescriptorSetLayout                           m_layout = nullptr;
	vk::DescriptorPool                                m_pool   = nullptr;
	vk::DescriptorSet                                 m_set    = nullptr;
	// Global map state; the map exists when GPU-SRT images are enabled.
	std::unique_ptr<Buffer>                           m_global_map;
	std::vector<GlobalBucket>                         m_global_buckets;
	std::vector<uint32_t>                             m_global_dirty; // Buckets to upload.
	std::deque<GlobalReadback>                        m_global_readbacks;
	std::vector<std::unique_ptr<Buffer>>              m_global_spare;
	std::map<std::array<uint32_t, 5>, uint32_t>       m_global_samplers; // S#, profile: index.
	std::vector<uint32_t>                             m_global_upload;
	uint64_t                                          m_global_generation = 0; // Map resets.
	uint32_t                                          m_global_known      = 0; // Resolved buckets.
	uint32_t                                          m_global_sampler_count = 0;
	bool m_global_clear = true;  // The whole device map is zeroed at the next commit.
	bool m_global_used  = false; // A commit since the last readback.

	friend struct ImageHeapTestAccess;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAP_H_
