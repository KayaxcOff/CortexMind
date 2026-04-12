//
// Created by muham on 12.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_BENCHMARK_UTILS_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_BENCHMARK_UTILS_HPP

#include <CortexMind/framework/Tools/params.hpp>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief Computes the arithmetic mean (average) of the values.
     * @param t Vector of measured times (in milliseconds)
     * @return Average value
     */
    [[nodiscard]]
    f64 compute_avg(const std::vector<f64>& t);
    /**
     * @brief Returns the minimum value in the vector.
     * @param t Vector of measured times
     * @return Minimum time
     */
    [[nodiscard]]
    f64 compute_min(const std::vector<f64>& t);
    /**
     * @brief Returns the maximum value in the vector.
     * @param t Vector of measured times
     * @return Maximum time
     */
    [[nodiscard]]
    f64 compute_max(const std::vector<f64>& t);
    /**
     * @brief Computes the median of the values.
     *
     * For even number of elements, returns the average of the two middle values.
     * Note: The vector is passed by value because it will be sorted internally.
     *
     * @param t Vector of measured times
     * @return Median time
     */
    [[nodiscard]]
    f64 compute_median(std::vector<f64> t);
    /**
     * @brief Computes the population standard deviation.
     * @param t   Vector of measured times
     * @param avg Pre-computed average (mean) of the times
     * @return Standard deviation
     */
    [[nodiscard]]
    f64 compute_std_dev(const std::vector<f64>& t, f64 avg);
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_BENCHMARK_UTILS_HPP