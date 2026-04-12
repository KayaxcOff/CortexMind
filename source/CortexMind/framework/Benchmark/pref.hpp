//
// Created by muham on 10.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_BENCHMARK_PREF_HPP
#define CORTEXMIND_FRAMEWORK_BENCHMARK_PREF_HPP

#include <CortexMind/framework/Benchmark/bench_output.hpp>
#include <functional>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief Lightweight benchmarking utility with warmup support and detailed statistics.
     */
    class PrefBench {
    public:
        /**
         * @brief Constructs a benchmark object.
         * @param name        Name of the benchmark
         * @param iterations  Number of measured iterations (default: 10)
         * @param warmup      Number of warmup iterations before measurement (default: 3)
         */
        explicit PrefBench(std::string name, i32 iterations = 10, i32 warmup = 3);
        ~PrefBench() = default;

        /**
         * @brief Executes warmup then measured iterations, records timings.
         * @param func Callable to benchmark
         */
        void run(const std::function<void()>& func);

        /**
         * @brief Prints detailed benchmark statistics to stdout.
         */
        void result() const;

        /**
         * @brief Returns the raw benchmark result struct.
         */
        [[nodiscard]]
        const BenchResult& get() const;

        /**
         * @brief Resets recorded times for reuse.
         */
        void reset();

    private:
        std::string name;
        i32 iterations;
        i32 warmup;
        std::vector<f64> times;
        BenchResult last_result{};
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_BENCHMARK_PREF_HPP