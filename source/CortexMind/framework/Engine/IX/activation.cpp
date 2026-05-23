//
// Created by muham on 21.05.2026.
//

#include "CortexMind/framework/Engine/IX/activation.hpp"
#include <CortexMind/framework/Engine/AVX2/activation.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/activation.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

void Activation::relu(const f32 *Xx, f32 *Xz, const size_t N, const DeviceType device) {
    CXM_ASSERT(Xx == nullptr, "Input pointer is null");
    CXM_ASSERT(Xz == nullptr, "Input pointer is null");
    CXM_ASSERT(N <= 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (device == DeviceType::kHOST) {
            avx2::Activation::relu(Xx, Xz, N);
        } else {
            cuda::Activation::relu(Xx, Xz, N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::relu(Xx, Xz, N);
    #endif //#if CXM_IS_CUDA_AVAILABLE
}

void Activation::tanh(const f32 *Xx, f32 *Xz, const size_t N, const DeviceType device) {
    CXM_ASSERT(Xx == nullptr, "Input pointer is null");
    CXM_ASSERT(Xz == nullptr, "Input pointer is null");
    CXM_ASSERT(N <= 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (device == DeviceType::kHOST) {
            avx2::Activation::tanh(Xx, Xz, N);
        } else {
            cuda::Activation::tanh(Xx, Xz, N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::tanh(Xx, Xz, N);
    #endif //#if CXM_IS_CUDA_AVAILABLE
}

void Activation::sigmoid(const f32 *Xx, f32 *Xz, const size_t N, const DeviceType device) {
    CXM_ASSERT(Xx == nullptr, "Input pointer is null");
    CXM_ASSERT(Xz == nullptr, "Input pointer is null");
    CXM_ASSERT(N <= 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (device == DeviceType::kHOST) {
            avx2::Activation::sigmoid(Xx, Xz, N);
        } else {
            cuda::Activation::sigmoid(Xx, Xz, N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::sigmoid(Xx, Xz, N);
    #endif //#if CXM_IS_CUDA_AVAILABLE
}

void Activation::sigmoid_fast(const f32 *Xx, f32 *Xz, const size_t N, const DeviceType device) {
    CXM_ASSERT(Xx == nullptr, "Input pointer is null");
    CXM_ASSERT(Xz == nullptr, "Input pointer is null");
    CXM_ASSERT(N <= 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (device == DeviceType::kHOST) {
            avx2::Activation::sigmoid_fast(Xx, Xz, N);
        } else {
            cuda::Activation::sigmoid_fast(Xx, Xz, N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::sigmoid_fast(Xx, Xz, N);
    #endif //#if CXM_IS_CUDA_AVAILABLE
}

void Activation::gelu(const f32 *Xx, f32 *Xz, const size_t N, const DeviceType device) {
    CXM_ASSERT(Xx == nullptr, "Input pointer is null");
    CXM_ASSERT(Xz == nullptr, "Input pointer is null");
    CXM_ASSERT(N <= 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (device == DeviceType::kHOST) {
            avx2::Activation::gelu(Xx, Xz, N);
        } else {
            cuda::Activation::gelu(Xx, Xz, N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::gelu(Xx, Xz, N);
    #endif //#if CXM_IS_CUDA_AVAILABLE
}
