#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <ranges>

namespace tune {
	using namespace std::literals;

	struct frame_info {
		std::vector<uint32_t> workload_distribution;
		std::chrono::steady_clock::duration duration;
		std::vector<std::chrono::steady_clock::duration> estimated_gpu_duration;
	};

	struct tuning_info {
		uint32_t total_workload;
		uint32_t gpu_count;

		std::vector<frame_info> frame_infos;
	};

	inline void init_tuning_info(tuning_info& info, uint32_t total_workload, uint32_t gpu_count) {
		info = {
			.total_workload = total_workload,
			.gpu_count = gpu_count
		};
	}

	inline void add_frame_info(tuning_info& info, frame_info frame) {
		if (info.frame_infos.size() > 10) {
			info.frame_infos = { info.frame_infos.back() };
		}
		info.frame_infos.push_back(frame);
	}

	inline std::optional<std::vector<uint32_t>> get_workload(tuning_info& info) {
		auto& frame_info = info.frame_infos.back();

		auto average_estimated_gpu_duration = std::accumulate(frame_info.estimated_gpu_duration.begin(), frame_info.estimated_gpu_duration.end(), 0ns)/frame_info.estimated_gpu_duration.size();
		auto variant_estimated_gpu_duration = 0.0;
		std::ranges::for_each(
			frame_info.estimated_gpu_duration,
			[&variant_estimated_gpu_duration, average_estimated_gpu_duration](auto duration) {
				auto v = (static_cast<float>(duration.count()) - average_estimated_gpu_duration.count()) / average_estimated_gpu_duration.count();
				variant_estimated_gpu_duration += v * v;
			}
		);
		if (variant_estimated_gpu_duration > 1.8) {
			const auto& estimated_gpu_duration = frame_info.estimated_gpu_duration;
			const auto& workload_distribution = frame_info.workload_distribution;
			float total_v = 0;
			auto gpu_v = std::vector<float>(estimated_gpu_duration.size());
			auto indices = std::vector<uint32_t>(gpu_v.size());
			std::ranges::iota(indices, 0);

			std::ranges::for_each(
				indices,
				[&estimated_gpu_duration, &workload_distribution,
				&total_v, &gpu_v](auto i) {
					auto v = static_cast<float>(workload_distribution[i]) / estimated_gpu_duration[i].count();
					total_v += v;
					gpu_v[i] = v;
				}
			);

			auto next_workload_distribution = std::vector<uint32_t>(workload_distribution.size());
			uint32_t remain = info.total_workload;
			std::ranges::for_each(
				indices,
				[&next_workload_distribution, &gpu_v, total_v, total_workload=info.total_workload, &remain](auto i) {
					next_workload_distribution[i] = static_cast<uint32_t>(total_workload * gpu_v[i] / total_v);
					remain -= next_workload_distribution[i];
				}
			);
			assert(remain < next_workload_distribution.size());
			for (int i = 0; i < remain; i++) {
				next_workload_distribution[i]++;
			}
			remain = 0;
			return next_workload_distribution;
		}
		else if (rand()%3) {
			auto best_frame_ite = std::ranges::min_element(info.frame_infos,
				std::ranges::less{},
				[](auto& frame_info) {
					return frame_info.duration;
				});
			auto next_workload_distribution = best_frame_ite->workload_distribution;

			auto extent_offset = std::vector<int32_t>(next_workload_distribution.size());
			auto dec_index = rand() % next_workload_distribution.size();
			auto inc_index = rand() % next_workload_distribution.size();
			if (next_workload_distribution[dec_index] > 1) {
				next_workload_distribution[inc_index] += 1;
				next_workload_distribution[dec_index] -= 1;
			}
			return next_workload_distribution;
		}
		else {
			return std::nullopt;
		}
	}
}