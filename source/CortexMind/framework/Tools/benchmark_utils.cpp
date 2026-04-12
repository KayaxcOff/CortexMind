//
// Created by muham on 12.04.2026.
//

#include "CortexMind/framework/Tools/benchmark_utils.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

using namespace cortex::_fw;

f64 cortex::_fw::compute_avg(const std::vector<f64> &t) {
    return std::accumulate(t.begin(), t.end(), 0.0) / static_cast<f64>(t.size());
}

f64 cortex::_fw::compute_min(const std::vector<f64> &t) {
    return *std::ranges::min_element(t);
}

f64 cortex::_fw::compute_max(const std::vector<f64> &t) {
    return *std::ranges::max_element(t);
}

f64 cortex::_fw::compute_median(std::vector<f64> t) {
    std::ranges::sort(t);
    const size_t mid = t.size() / 2;
    return (t.size() % 2 == 0) ? (t[mid - 1] + t[mid]) / 2.0 : t[mid];
}

f64 cortex::_fw::compute_std_dev(const std::vector<f64> &t, const f64 avg) {
    f64 acc = 0.0;
    for (const f64 v : t) {
        acc += (v - avg) * (v - avg);
    }
    return std::sqrt(acc / static_cast<f64>(t.size()));
}
