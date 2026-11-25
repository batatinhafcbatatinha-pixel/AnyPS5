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
	    : m_ranges(std::move(other.m_ranges)), m_recent(other.m_recent),
	      m_recent_next(other.m_recent_next) {
		other.Reset();
	}
	RangeSet& operator=(RangeSet&& other) noexcept {
		if (this != &other) {
			m_ranges      = std::move(other.m_ranges);
			m_recent      = other.m_recent;
			m_recent_next = other.m_recent_next;
			other.Reset();
		}
		return *this;
	}

	void Add(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		// The same ranges are added again and again (a buffer every draw writes): one inside a
		// range added lately is in the set already.
		for (const auto& [begin, last]: m_recent) {
			if (begin <= address && end <= last) {
				return;
			}
		}
		auto it = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second >= address) {
			it = std::prev(it);
		}
		if (it == m_ranges.end() || it->first > address) {
			it = m_ranges.emplace_hint(it, address, end);
		} else {
			if (it->second >= end) {
				Remember(it->first, it->second);
				return;
			}
			it->second = end;
		}
		auto next = std::next(it);
		while (next != m_ranges.end() && next->first <= it->second) {
			it->second = std::max(it->second, next->second);
			next       = m_ranges.erase(next);
		}
		Remember(it->first, it->second);
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
		while (it != m_ranges.end() && it->first < end) {
			const auto begin = it->first;
			const auto last  = it->second;
			Forget();
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
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			return true;
		}
		return it != m_ranges.end() && it->first < end;
	}

	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
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

	// Ranges each inside one range of the set; a removal forgets them. Empty slots are {0, 0}.
	void Remember(uint64_t begin, uint64_t end) {
		m_recent[m_recent_next] = {begin, end};
		m_recent_next           = (m_recent_next + 1) % m_recent.size();
	}
	void Forget() {
		m_recent      = {};
		m_recent_next = 0;
	}
	void Reset() noexcept {
		m_ranges.clear();
		m_recent      = {};
		m_recent_next = 0;
	}

	std::map<uint64_t, uint64_t>                 m_ranges;
	std::array<std::pair<uint64_t, uint64_t>, 4> m_recent {};
	size_t                                       m_recent_next = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
