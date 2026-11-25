#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_

#include "common/assert.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <map>
#include <utility>

namespace Libs::Graphics {

class RangeSet final {
public:
	RangeSet()                           = default;
	RangeSet(const RangeSet&)            = default;
	RangeSet& operator=(const RangeSet&) = default;
	// A moved-from set is empty, so its recent ranges go with the ranges.
	RangeSet(RangeSet&& other) noexcept
	    : m_ranges(std::move(other.m_ranges)), m_recent(other.m_recent) {
		other.Reset();
	}
	RangeSet& operator=(RangeSet&& other) noexcept {
		if (this != &other) {
			m_ranges = std::move(other.m_ranges);
			m_recent = other.m_recent;
			other.Reset();
		}
		return *this;
	}

	void Add(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		// The same ranges are added again and again (a buffer every draw writes, the spans of a
		// table every dispatch writes): one inside a range added lately is in the set already.
		if (Recalled(address, end)) {
			return;
		}
		auto it = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second >= address) {
			it = std::prev(it);
		}
		if (it == m_ranges.end() || it->first > address) {
			it = m_ranges.emplace_hint(it, address, end);
		} else {
			if (it->second >= end) {
				Remember(address, it->first, it->second);
				return;
			}
			it->second = end;
		}
		auto next = std::next(it);
		while (next != m_ranges.end() && next->first <= it->second) {
			it->second = std::max(it->second, next->second);
			next       = m_ranges.erase(next);
		}
		Remember(address, it->first, it->second);
	}

	void Subtract(uint64_t address, uint64_t size) {
		Extract(address, size, [](uint64_t, uint64_t) {});
	}

	// ForEachInRange followed by Subtract of the same range, in one walk.
	template <typename Func>
	void Extract(uint64_t address, uint64_t size, Func&& func) {
		const auto end = End(address, size);
		if (m_ranges.empty()) {
			return;
		}
		auto it = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			it = std::prev(it);
		}
		if (it != m_ranges.end() && it->first < end) {
			Forget();
		}
		while (it != m_ranges.end() && it->first < end) {
			const auto begin = it->first;
			const auto last  = it->second;
			func(std::max(begin, address), std::min(last, end));
			it = m_ranges.erase(it);
			if (begin < address) {
				m_ranges.emplace_hint(it, begin, address);
			}
			if (last > end) {
				m_ranges.emplace_hint(it, end, last);
				break;
			}
		}
	}

	template <typename Func>
	void ForEach(Func&& func) const {
		for (const auto& [begin, end]: m_ranges) {
			func(begin, end);
		}
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		if (Recalled(address, end)) {
			return true;
		}
		auto it = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			return true;
		}
		return it != m_ranges.end() && it->first < end;
	}

	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		if (Recalled(address, end)) {
			return true;
		}
		auto it = m_ranges.upper_bound(address);
		if (it == m_ranges.begin()) {
			return false;
		}
		--it;
		return it->first <= address && it->second >= end;
	}

	template <typename Func>
	void ForEachInRange(uint64_t address, uint64_t size, Func&& func) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (; it != m_ranges.end() && it->first < end; ++it) {
			const auto begin = std::max(address, it->first);
			const auto last  = std::min(end, it->second);
			if (begin < last) {
				func(begin, last);
			}
		}
	}

	[[nodiscard]] bool Empty() const { return m_ranges.empty(); }

private:
	static uint64_t End(uint64_t address, uint64_t size) {
		if (size == 0 || size > UINT64_MAX - address) {
			EXIT("invalid range-set address or size\n");
		}
		return address + size;
	}

	// Ranges each inside one range of the set, in a slot chosen by the address of the Add that
	// found them: an Add only grows ranges, so they stay inside, and a removal forgets them all.
	// Repeated adds of up to a few dozen distinct ranges (the written spans of one dispatch) are
	// answered here without a tree walk. Empty slots are {0, 0}, which hold no nonempty range.
	static constexpr unsigned RECENT_BITS  = 5;
	static constexpr size_t   RECENT_SLOTS = size_t {1} << RECENT_BITS;
	[[nodiscard]] static size_t Slot(uint64_t address) noexcept {
		return static_cast<size_t>(((address >> 4u) * 0x9e3779b97f4a7c15ull) >> (64u - RECENT_BITS));
	}
	[[nodiscard]] bool Recalled(uint64_t address, uint64_t end) const noexcept {
		const auto& [begin, last] = m_recent[Slot(address)];
		return begin <= address && end <= last;
	}
	void Remember(uint64_t address, uint64_t begin, uint64_t end) noexcept {
		m_recent[Slot(address)] = {begin, end};
	}
	void Forget() noexcept { m_recent = {}; }
	void Reset() noexcept {
		m_ranges.clear();
		m_recent = {};
	}

	std::map<uint64_t, uint64_t>                            m_ranges;
	std::array<std::pair<uint64_t, uint64_t>, RECENT_SLOTS> m_recent {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
