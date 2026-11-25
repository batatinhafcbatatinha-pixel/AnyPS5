#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_HOTPAGES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_HOTPAGES_H_

#include "graphics/host_gpu/regionDefinitions.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace Libs::Graphics {

// Pages the guest CPU rewrites in nearly every frame, kept unwatched by the buffer tracker
// (RegionManager's exact hot pages). Without a write watch no fault reports their writes, so
// each keeps a shadow: the bytes its cache buffer received at its last upload. A consumer that
// would have asked whether the page is CPU-modified compares the guest bytes with the shadow
// instead, so it sees every write that a fault would have reported by then.
//
// Slots are taken on the GPU thread only (promotion happens in its uploads). A demotion may run
// on any thread under the page's region lock and only releases the slot; released slots are
// reused after Recycle, which the GPU thread runs at the start of its tracker uploads, so an
// upload that still copies into the shadow of a page demoted meanwhile writes an unused slot.
class HotPages final {
public:
	static constexpr uint16_t NO_SLOT  = 0xffff;
	static constexpr size_t   CAPACITY = 1024; // 4 MiB of shadows
	static_assert(CAPACITY < NO_SLOT);

	// The guest bytes of a page through a view no page protection covers, or nullptr when the
	// page has none (then it is not kept hot).
	using PageReader = const uint8_t* (*)(uint64_t page);

	HotPages() = default;
	KYTY_CLASS_NO_COPY(HotPages);

	void                     SetReader(PageReader reader) noexcept { m_reader = reader; }
	[[nodiscard]] const uint8_t* Read(uint64_t page) const noexcept {
		return m_reader != nullptr ? m_reader(page) : nullptr;
	}

	// Pages kept hot now (released slots excluded).
	[[nodiscard]] uint32_t Live() const noexcept { return m_live.load(std::memory_order_acquire); }

	// GPU thread: false while promotion is suspended (see EndFrame).
	[[nodiscard]] bool PromotionAllowed(uint64_t frame) const noexcept {
		return m_reader != nullptr && frame >= m_suspended_until;
	}

	// GPU thread: a slot for `page`, its shadow not valid yet; NO_SLOT when all are taken.
	[[nodiscard]] uint16_t Acquire(uint64_t page, uint64_t frame) {
		uint16_t slot = NO_SLOT;
		if (!m_free.empty()) {
			slot = m_free.back();
			m_free.pop_back();
		} else if (m_used < CAPACITY) {
			if (m_shadows == nullptr) {
				m_shadows = std::make_unique<uint8_t[]>(CAPACITY * TRACKER_PAGE_SIZE);
			}
			slot = static_cast<uint16_t>(m_used++);
		} else {
			return NO_SLOT;
		}
		auto& entry = m_slots[slot];
		entry.valid.store(false, std::memory_order_relaxed);
		entry.deferred      = false;
		entry.changed_frame = frame;
		entry.page.store(page, std::memory_order_release);
		m_live.fetch_add(1, std::memory_order_acq_rel);
		return slot;
	}

	// Any thread, under the page's region lock: the page is watched normally again.
	void Release(uint16_t slot) {
		m_slots[slot].page.store(0, std::memory_order_release);
		m_live.fetch_sub(1, std::memory_order_acq_rel);
		std::lock_guard lock(m_release_mutex);
		m_released.push_back(slot);
		m_any_released.store(true, std::memory_order_release);
	}

	// GPU thread, when no upload of an earlier tracker call is still copying (see above).
	void Recycle() {
		if (!m_any_released.load(std::memory_order_acquire)) {
			return;
		}
		std::lock_guard lock(m_release_mutex);
		m_free.insert(m_free.end(), m_released.begin(), m_released.end());
		m_released.clear();
		m_any_released.store(false, std::memory_order_release);
	}

	// GPU thread: whether the guest bytes of the slot's page differ from the bytes its buffer
	// holds. A shadow not filled by a whole-page upload, or a page without a reader view,
	// counts as changed.
	[[nodiscard]] bool Changed(uint16_t slot, uint64_t page, uint64_t frame) {
		auto& entry = m_slots[slot];
		if (entry.deferred) {
			// An upload recorded earlier in this batch copies the page's bytes as they are when
			// the batch is submitted (BufferCache::FlushDeferredTracking): every use recorded
			// after it reads them, so the page is current for it.
			return false;
		}
		m_compares++;
		const auto* bytes = Read(page);
		if (!entry.valid.load(std::memory_order_acquire) || bytes == nullptr ||
		    std::memcmp(bytes, Shadow(slot), TRACKER_PAGE_SIZE) != 0) {
			entry.changed_frame = frame;
			return true;
		}
		return false;
	}

	// GPU thread: copies `size` guest bytes at `address` (inside the slot's page) into the
	// shadow, then from the shadow to `destination`, so the buffer gets exactly the shadow's
	// bytes. False, with nothing copied, when the page has no reader view.
	[[nodiscard]] bool Refresh(uint16_t slot, uint64_t address, uint64_t size, uint8_t* destination) {
		const auto  page   = address & ~(TRACKER_PAGE_SIZE - 1);
		const auto* bytes  = Read(page);
		auto&       entry  = m_slots[slot];
		if (bytes == nullptr) {
			entry.valid.store(false, std::memory_order_release);
			return false;
		}
		const auto offset = address - page;
		auto*      shadow = Shadow(slot) + offset;
		std::memcpy(shadow, bytes + offset, size);
		std::memcpy(destination, shadow, size);
		// Part of a page leaves the rest of the shadow unknown: compared as changed.
		entry.valid.store(size == TRACKER_PAGE_SIZE, std::memory_order_release);
		return true;
	}

	// GPU thread: an upload of the slot's page waits for the deferred flush (see Changed).
	void MarkDeferred(uint16_t slot) noexcept {
		m_slots[slot].deferred = true;
		m_any_deferred         = true;
	}
	// GPU thread, after the deferred uploads were filled.
	void ClearDeferred() noexcept {
		if (!m_any_deferred) {
			return;
		}
		for (size_t slot = 0; slot < m_used; slot++) {
			m_slots[slot].deferred = false;
		}
		m_any_deferred = false;
	}

	// GPU thread: for each live slot, func(slot, page).
	template <typename Func>
	void ForEachLive(Func&& func) const {
		for (size_t slot = 0; slot < m_used; slot++) {
			const auto page = m_slots[slot].page.load(std::memory_order_acquire);
			if (page != 0) {
				func(static_cast<uint16_t>(slot), page);
			}
		}
	}

	[[nodiscard]] uint64_t ChangedFrame(uint16_t slot) const noexcept {
		return m_slots[slot].changed_frame;
	}

	// GPU thread, once per frame seen: whether the compares of the frame that ended cost more
	// than the protection changes and faults the hot pages saved. Comparing an unchanged page
	// reads 8 KiB; a page kept watched costs a fault and two protection changes per frame,
	// about the time of a few dozen compares. Over budget, promotion stops for a while and the
	// caller demotes every hot page. APS5_KYTY_HOT_COMPARES=n sets compares allowed per hot page.
	[[nodiscard]] bool EndFrame(uint64_t frame) {
		if (frame == m_frame) {
			return false;
		}
		static const uint64_t per_page = [] {
			const char* value = std::getenv("APS5_KYTY_HOT_COMPARES");
			const auto  given = value != nullptr ? std::strtoull(value, nullptr, 10) : 0;
			return given != 0 ? given : uint64_t {24};
		}();
		const bool over = m_compares > uint64_t {1024} + per_page * m_live.load(std::memory_order_acquire);
		m_compares      = 0;
		m_frame         = frame;
		if (over) {
			m_suspended_until = frame + 64;
		}
		return over;
	}

private:
	struct Slot {
		std::atomic<uint64_t> page {0}; // 0: free or released
		std::atomic<bool>     valid {false};
		uint64_t              changed_frame = 0;     // GPU thread
		bool                  deferred      = false; // GPU thread (MarkDeferred)
	};

	[[nodiscard]] uint8_t* Shadow(uint16_t slot) noexcept {
		return m_shadows.get() + size_t {slot} * TRACKER_PAGE_SIZE;
	}

	PageReader                   m_reader = nullptr;
	std::array<Slot, CAPACITY>   m_slots {};
	std::unique_ptr<uint8_t[]>   m_shadows;
	size_t                       m_used = 0;          // slots ever taken (GPU thread)
	std::vector<uint16_t>        m_free;              // GPU thread
	std::mutex                   m_release_mutex;
	std::vector<uint16_t>        m_released;          // guarded by m_release_mutex
	std::atomic<bool>            m_any_released {false};
	std::atomic<uint32_t>        m_live {0};
	bool                         m_any_deferred    = false; // GPU thread
	uint64_t                     m_compares        = 0; // GPU thread, this frame
	uint64_t                     m_frame           = 0; // GPU thread
	uint64_t                     m_suspended_until = 0; // GPU thread
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_HOTPAGES_H_
