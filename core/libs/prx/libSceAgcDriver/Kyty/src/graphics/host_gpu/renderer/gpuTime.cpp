#include "graphics/host_gpu/renderer/gpuTime.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::GpuTime {
namespace {

constexpr uint32_t QueryCount = 1u << 16u;

struct Total {
	uint64_t ns    = 0;
	uint64_t count = 0;
	char     kind  = 0;
};

struct State {
	vk::Device                              device;
	vk::QueryPool                           pool;
	double                                  period_ns  = 1.0;
	uint64_t                                valid_mask = ~uint64_t {0};
	uint64_t                                allocated  = 0; // queries handed out
	uint64_t                                freed      = 0; // queries read and reset, in order
	uint64_t                                batch_tick = 0; // 0: no open batch
	bool                                    batch_closed = false; // out of queries: no more in this batch
	uint64_t                                batch_first = 0;
	std::vector<std::pair<uint64_t, char>>  batch_keys;
	std::unordered_map<uint64_t, Total>     totals;
	uint64_t                                total_ns = 0;
	uint64_t                                ops      = 0;
	uint64_t                                skipped  = 0;
	std::chrono::steady_clock::time_point   last_report = std::chrono::steady_clock::now();
};

State* g_state = nullptr;

State* Get(RenderContext& context) {
	if (g_state != nullptr) return g_state;
	auto& graphics = context.GetGraphics();
	auto* state    = new State;
	state->device  = graphics.device;
	const vk::QueryPoolCreateInfo info {.queryType = vk::QueryType::eTimestamp, .queryCount = QueryCount};
	RequireVulkanSuccess(state->device.createQueryPool(&info, nullptr, &state->pool), "vkCreateQueryPool");
	state->device.resetQueryPool(state->pool, 0, QueryCount);
	state->period_ns = graphics.physical_device_properties.limits.timestampPeriod;
	const auto bits  = graphics.timestamp_valid_bits;
	state->valid_mask = bits >= 64 ? ~uint64_t {0} : (uint64_t {1} << bits) - 1;
	g_state           = state;
	return state;
}

bool Alloc(State& state, uint32_t& query) {
	if (state.allocated - state.freed >= QueryCount) {
		state.skipped++;
		return false;
	}
	query = static_cast<uint32_t>(state.allocated++ % QueryCount);
	return true;
}

void Report(State& state) {
	const auto now = std::chrono::steady_clock::now();
	if (now - state.last_report < std::chrono::seconds(10)) return;
	state.last_report = now;
	std::vector<std::pair<uint64_t, Total>> rows(state.totals.begin(), state.totals.end());
	std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
	char head[160];
	std::snprintf(head, sizeof(head), "[gputime] 10 s: %.0f ms GPU time in %llu draws/dispatches (%llu not timed); top:",
	              static_cast<double>(state.total_ns) / 1e6, static_cast<unsigned long long>(state.ops),
	              static_cast<unsigned long long>(state.skipped));
	std::string text = head;
	for (size_t index = 0; index < rows.size() && index < 24; ++index) {
		char row[96];
		std::snprintf(row, sizeof(row), " %c:%016llx %.1fms/%llu", rows[index].second.kind,
		              static_cast<unsigned long long>(rows[index].first),
		              static_cast<double>(rows[index].second.ns) / 1e6,
		              static_cast<unsigned long long>(rows[index].second.count));
		text += row;
	}
	Log::WriteToConsoleAndLog(text + std::string(1, char(10)));
	state.totals.clear();
	state.total_ns = 0;
	state.ops      = 0;
	state.skipped  = 0;
}

// Runs once the batch completed (a deferred operation, GPU thread): adds its times, frees its queries.
void Read(State& state, uint64_t first, std::vector<std::pair<uint64_t, char>>& keys) {
	const auto           count = static_cast<uint32_t>(keys.size());
	std::vector<uint64_t> stamps(count);
	bool                  ok    = true;
	for (uint32_t done = 0; done < count;) {
		const auto slot  = static_cast<uint32_t>((first + done) % QueryCount);
		const auto chunk = std::min(count - done, QueryCount - slot);
		const auto result = state.device.getQueryPoolResults(state.pool, slot, chunk, chunk * sizeof(uint64_t),
		                                                     stamps.data() + done, sizeof(uint64_t),
		                                                     vk::QueryResultFlagBits::e64);
		ok &= result == vk::Result::eSuccess;
		state.device.resetQueryPool(state.pool, slot, chunk);
		done += chunk;
	}
	state.freed += count;
	if (ok) {
		for (uint32_t index = 1; index < count; ++index) {
			const auto ticks = (stamps[index] - stamps[index - 1]) & state.valid_mask;
			const auto ns    = static_cast<uint64_t>(static_cast<double>(ticks) * state.period_ns);
			auto&      total = state.totals[keys[index].first];
			total.ns += ns;
			total.count++;
			total.kind = keys[index].second;
			state.total_ns += ns;
			state.ops++;
		}
	}
	Report(state);
}

void Flush(State& state, CommandScheduler& scheduler) {
	if (state.batch_tick == 0) return;
	state.batch_tick = 0;
	scheduler.DeferOperation([&state, first = state.batch_first, keys = std::move(state.batch_keys)]() mutable {
		Read(state, first, keys);
	});
	state.batch_keys = {};
}

} // namespace

bool Enabled(RenderContext& context) {
	static const bool wanted = std::getenv("APS5_KYTY_GPUTIME") != nullptr;
	return wanted && context.GetGraphics().gpu_time_enabled;
}

void Before(RenderContext& context, vk::CommandBuffer command) {
	auto& scheduler = context.GetCommandScheduler();
	if (scheduler.InSide()) return;
	auto&      state = *Get(context);
	const auto tick  = scheduler.CurrentTick();
	if (tick == state.batch_tick) return;
	Flush(state, scheduler);
	uint32_t query = 0;
	if (!Alloc(state, query)) return;
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, state.pool, query);
	state.batch_tick  = tick;
	state.batch_closed = false;
	state.batch_first = state.allocated - 1;
	state.batch_keys.push_back({0, 's'});
}

void After(RenderContext& context, vk::CommandBuffer command, uint64_t key, char kind) {
	auto& scheduler = context.GetCommandScheduler();
	if (scheduler.InSide()) return;
	auto& state = *Get(context);
	// A submit in the middle (a stream buffer wrap) left the opening timestamp in the previous batch.
	if (state.batch_tick != scheduler.CurrentTick() || state.batch_closed) return;
	uint32_t query = 0;
	if (!Alloc(state, query)) {
		state.batch_closed = true; // the batch's queries stay contiguous
		return;
	}
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, state.pool, query);
	state.batch_keys.push_back({key, kind});
}

} // namespace Libs::Graphics::GpuTime
