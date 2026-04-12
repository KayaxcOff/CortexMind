//
// Created by muham on 10.04.2026.
//

#include "CortexMind/framework/Benchmark/pref.hpp"
#include <CortexMind/framework/Tools/benchmark_utils.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <utility>

using namespace cortex::_fw;

PrefBench::PrefBench(std::string name, const i32 iterations, const i32 warmup) : name(std::move(name)), iterations(iterations), warmup(warmup) {
    this->times.reserve(iterations);
}

void PrefBench::run(const std::function<void()>& func) {
    this->times.clear();

    for (i32 i = 0; i < this->warmup; ++i) {
        func();
    }

    for (i32 i = 0; i < this->iterations; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        func();
        const auto end = std::chrono::high_resolution_clock::now();
        this->times.push_back(std::chrono::duration<f64, std::milli>(end - start).count());
    }

    const f64 avg = compute_avg(this->times);
    this->last_result = {
        avg,
        compute_min(this->times),
        compute_max(this->times),
        compute_median(this->times),
        compute_std_dev(this->times, avg),
        this->iterations,
        this->name
    };
}

void PrefBench::result() const {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== Benchmark: " << this->last_result.name << " ===\n";
    std::cout << "  Iterations : " << this->last_result.iterations << "\n";
    std::cout << "  Avg        : " << this->last_result.avg     << " ms\n";
    std::cout << "  Min        : " << this->last_result.min     << " ms\n";
    std::cout << "  Max        : " << this->last_result.max     << " ms\n";
    std::cout << "  Median     : " << this->last_result.median  << " ms\n";
    std::cout << "  Std Dev    : " << this->last_result.std_dev << " ms\n";
}

const BenchResult& PrefBench::get() const {
    return this->last_result;
}

void PrefBench::reset() {
    this->times.clear();
    this->last_result = {};
}