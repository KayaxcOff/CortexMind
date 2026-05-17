//
// Created by muham on 16.05.2026.
//

#include "CortexMind/framework/Engine/IX/fill.hpp"
#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/Kernels/activation.cuh>
    #include <CortexMind/framework/Tools/cuda.cuh>
    #include <CortexMind/framework/Tools/kernel_operations.hpp>
#endif
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

void FillOp::fill(TensorStorage* x, const f32 value, const size_t N) {
    CXM_ASSERT(!x->isValid(), "Storage is null");
    CXM_ASSERT(N == 0,        "N must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (x->device() == DeviceType::kHOST) {
            f32* ptr = x->data();
            const avx2::vec8f val = avx2::set1(value);
            size_t i = 0;
            for (; i + 8 <= N; i += 8) avx2::storeu(ptr + i, val);
            for (; i < N; ++i) ptr[i] = value;
        } else {
            const auto* Xx4 = reinterpret_cast<const cuda::f32x4*>(x->data());
            auto*       Xz4 = reinterpret_cast<cuda::f32x4*>(x->data());
            cuda::kernels::activation<<<cuda::grid1d(N >> 2), cuda::BLOCK_SIZE_1D>>>(
                Xx4, Xz4, N, ops::Constant{value});
        }
    #else
        f32* ptr = x->data();
        const avx2::vec8f val = avx2::set1(value);
        size_t i = 0;
        for (; i + 8 <= N; i += 8) avx2::storeu(ptr + i, val);
        for (; i < N; ++i) ptr[i] = value;
    #endif
}

void FillOp::zero(TensorStorage* x, const size_t N) {
    CXM_ASSERT(!x->isValid(), "Storage is null");
    CXM_ASSERT(N == 0,        "N must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (x->device() == DeviceType::kHOST) {
            f32* ptr = x->data();
            const avx2::vec8f val = avx2::zero();
            size_t i = 0;
            for (; i + 8 <= N; i += 8) avx2::storeu(ptr + i, val);
            for (; i < N; ++i) ptr[i] = 0.0f;
        } else {
            CXM_CUDA_ASSERT(cuda::memset<f32>(x->data(), 0.0f, N));
        }
    #else
        f32* ptr = x->data();
        const avx2::vec8f val = avx2::zero();
        size_t i = 0;
        for (; i + 8 <= N; i += 8) avx2::storeu(ptr + i, val);
        for (; i < N; ++i) ptr[i] = 0.0f;
    #endif
}

void FillOp::ones(TensorStorage* x, const size_t N) {
    CXM_ASSERT(!x->isValid(), "Storage is null");
    CXM_ASSERT(N == 0,        "N must be non-zero");

    // ones = fill(1.0f)
    FillOp::fill(x, 1.0f, N);
}