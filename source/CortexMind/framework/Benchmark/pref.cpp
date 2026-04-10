//
// Created by muham on 10.04.2026.
//

#include "CortexMind/framework/Benchmark/pref.hpp"
#include <iostream>
#include <utility>
#include <numeric>

using namespace cortex::_fw;

PrefBench::PrefBench(std::string name, const i32 iterations) : name(std::move(name)), iterations(iterations) {}

PrefBench::~PrefBench() = default;

void PrefBench::run(const std::function<void()> &func) {
    for (i32 i = 0; i < this->iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        func();

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        this->times.push_back(duration.count());
    }
}

void PrefBench::result() const {
    std::cout << "Benchmark result of " << this->name << " in " << this->iterations << "iterations" << std::endl;
    std::cout << average(this->times) << std::endl;
}

f64 PrefBench::average(const std::vector<f64> &times) {
    const double sum = std::accumulate(times.begin(), times.end(), 0.0);
    return sum / static_cast<f64>(times.size());
}
