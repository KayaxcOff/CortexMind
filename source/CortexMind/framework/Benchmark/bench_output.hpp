//
// Created by muham on 12.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_BENCH_OUTPUT_HPP
#define CORTEXMIND_FRAMEWORK_BENCH_OUTPUT_HPP

#include <CortexMind/framework/Tools/params.hpp>
#include <string>

namespace cortex::_fw {
    /**
     * @brief Result of a benchmark run.
     */
    struct BenchResult {
        f64 avg;
        f64 min;
        f64 max;
        f64 median;
        f64 std_dev;
        i32 iterations;
        std::string name;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_BENCH_OUTPUT_HPP