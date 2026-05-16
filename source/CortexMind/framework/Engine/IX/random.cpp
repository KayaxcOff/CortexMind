//
// Created by muham on 16.05.2026.
//

#include "CortexMind/framework/Engine/IX/random.hpp"
#include <CortexMind/framework/Tools/err.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/scalar.cuh>
    #include <CortexMind/runtime/curand.cuh>
    #include <curand.h>
#endif
#include <random>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

void RandomOp::uniform(TensorStorage* x, const f32 min, const f32 max, const size_t N) {
    CXM_ASSERT(!x->isValid(), "Storage is null");
    CXM_ASSERT(N == 0,        "N must be non-zero");
    CXM_ASSERT(min >= max,    "min must be less than max");

    #if CXM_IS_CUDA_AVAILABLE
        if (x->device() == DeviceType::kHOST) {
            static thread_local std::mt19937 rng{std::random_device{}()};
            std::uniform_real_distribution<f32> dist(min, max);
            f32* ptr = x->data();
            for (size_t i = 0; i < N; ++i) ptr[i] = dist(rng);
        } else {
            // cuRAND [0, 1) üretir, sonra [min, max]'a scale ederiz
            CXM_ASSERT(
                curandGenerateUniform(runtime::Curand::instance().rand_handle, x->data(), N)
                != CURAND_STATUS_SUCCESS,
                "curandGenerateUniform() failed"
            );

            // [0,1) → [min, max]: x = x * (max - min) + min
            // ScalarOp kullanarak scale + shift
            const f32 range = max - min;
            if (range != 1.0f) {
                cuda::ScalarKernel::mul(x->data(), range, N);
            }
            if (min != 0.0f) {
                cuda::ScalarKernel::add(x->data(), min, N);
            }
        }
    #else
        static thread_local std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<f32> dist(min, max);
        f32* ptr = x->data();
        for (size_t i = 0; i < N; ++i) ptr[i] = dist(rng);
    #endif
}

void RandomOp::normal(TensorStorage* x, const f32 mean, const f32 std, const size_t N) {
    CXM_ASSERT(!x->isValid(), "Storage is null");
    CXM_ASSERT(N == 0,        "N must be non-zero");
    CXM_ASSERT(std <= 0.0f,   "Standard deviation must be positive");

    #if CXM_IS_CUDA_AVAILABLE
        if (x->device() == DeviceType::kHOST) {
            static thread_local std::mt19937 rng{std::random_device{}()};
            std::normal_distribution<f32> dist(mean, std);
            f32* ptr = x->data();
            for (size_t i = 0; i < N; ++i) ptr[i] = dist(rng);
        } else {
            // cuRAND normal: mean=0, std=1 üretir
            // N çift sayı olmalı — curandGenerateNormal için
            const size_t aligned_N = (N % 2 == 0) ? N : N + 1;

            CXM_ASSERT(
                curandGenerateNormal(runtime::Curand::instance().rand_handle,
                    x->data(), aligned_N, mean, std)
                != CURAND_STATUS_SUCCESS,
                "curandGenerateNormal() failed"
            );
        }
    #else
        static thread_local std::mt19937 rng{std::random_device{}()};
        std::normal_distribution<f32> dist(mean, std);
        f32* ptr = x->data();
        for (size_t i = 0; i < N; ++i) ptr[i] = dist(rng);
    #endif
}