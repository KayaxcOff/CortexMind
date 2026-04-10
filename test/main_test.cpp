//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <CortexMind/core/Engine/AVX2/matrix.hpp>
#include <CortexMind/framework/Benchmark/pref.hpp>
#include <vector>
#include <random>

using namespace cortex::_fw;

int main() {
    constexpr size_t N = 512;

    std::vector<f32> a(N * N);
    std::vector<f32> b(N * N);
    std::vector<f32> c(N * N);

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution dist(0.0f, 1.0f);

    for (size_t i = 0; i < N; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    PrefBench bench("matmul-test", 10);

    bench.run([&]() {
        avx2::matrix_t::matmul(a.data(), b.data(), c.data(), N, N, N);
    });
    bench.result();

    return 0;
}

// Output:
//Benchmark result of matmul-test in 10 iterations
//101.739
// matrix_t::matmul() needs a upgrade