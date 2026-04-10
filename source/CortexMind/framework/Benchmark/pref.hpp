//
// Created by muham on 10.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_BENCHMARK_PREF_HPP
#define CORTEXMIND_FRAMEWORK_BENCHMARK_PREF_HPP

#include <CortexMind/framework/Tools/params.hpp>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief Simple and lightweight benchmarking utility for performance measurements.
     *
     * Measures the execution time of a given function over multiple iterations
     * and reports the average time in milliseconds.
     */
    class PrefBench {
    public:
        /**
         * @brief Constructs a benchmark object.
         * @param name        Name of the benchmark (will be printed in results)
         * @param iterations  Number of times the function will be executed (default: 10)
         */
        explicit PrefBench(std::string  name, i32 iterations = 10);
        ~PrefBench();

        /**
         * @brief Executes the given function multiple times and records timings.
         * @param func Callable object (lambda, function pointer, etc.) to benchmark
         */
        void run(const std::function<void()>& func);
        /**
         * @brief Prints the benchmark name and average execution time to stdout.
         */
        void result() const;

    private:
        std::string name;
        i32 iterations;
        std::vector<f64> times;

        /**
         * @brief Calculates the average of recorded times.
         * @param times Vector containing measured durations in milliseconds
         * @return Average time in milliseconds
         */
        [[nodiscard]]
        static f64 average(const std::vector<f64>& times);
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_BENCHMARK_PREF_HPP