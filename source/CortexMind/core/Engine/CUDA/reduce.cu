//
// Created by muham on 9.04.2026.
//

#include "CortexMind/core/Engine/CUDA/reduce.h"
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/runtime/ctx.h>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

ReduceOp::ReduceOp() {
    CXM_CUDA_ASSERT(
        cudaHostAlloc(&this->host_output, sizeof(f32), cudaHostAllocMapped),
        "ReduceOp::ReduceOp()"
    );
    CXM_CUDA_ASSERT(
        cudaHostGetDevicePointer(&this->cuda_output, this->host_output, 0),
        "ReduceOp::ReduceOp()"
    );
    CXM_CUDA_ASSERT(
        cudaHostAlloc(&this->host_index, sizeof(i32), cudaHostAllocMapped),
        "ReduceOp::ReduceOp()"
    );
    CXM_CUDA_ASSERT(
        cudaHostGetDevicePointer(&this->cuda_index, this->host_index, 0),
        "ReduceOp::ReduceOp()"
    );
}

ReduceOp::~ReduceOp() {
    cudaFreeHost(this->host_output);
    cudaFreeHost(this->host_index);
}

f32 ReduceOp::dot(const f32* Xx, const f32* Xy, const size_t N) {
    cublasSdot(
        runtime::CublasContext::instance().handle,
        static_cast<int>(N),
        Xx, 1,
        Xy, 1,
        this->cuda_output
    );
    cuda::DeviceSynchronize();
    return *this->host_output;
}

f32 ReduceOp::norm2(const f32* x, const size_t N) {
    cublasSnrm2(
        runtime::CublasContext::instance().handle,
        static_cast<int>(N),
        x, 1,
        this->cuda_output
    );
    cuda::DeviceSynchronize();
    return *this->host_output;
}

f32 ReduceOp::norm1(const f32* x, const size_t N) {
    cublasSasum(
        runtime::CublasContext::instance().handle,
        static_cast<int>(N),
        x, 1,
        this->cuda_output
    );
    cuda::DeviceSynchronize();
    return *this->host_output;
}

f32 ReduceOp::min(const f32* x, const size_t N) {
    cublasIsamin(
        runtime::CublasContext::instance().handle,
        static_cast<int>(N),
        x, 1,
        this->cuda_index
    );
    cudaDeviceSynchronize();
    const i32 idx = *this->host_index - 1;
    f32 result;
    cuda::memcpy(&result, x + idx, sizeof(f32), CXM_DEVICE_TO_HOST);
    return result;
}

f32 ReduceOp::max(const f32* x, const size_t N) {
    cublasIsamax(
        runtime::CublasContext::instance().handle,
        static_cast<int>(N),
        x, 1,
        this->cuda_index
    );
    cuda::DeviceSynchronize();
    const i32 idx = *this->host_index - 1;
    f32 result;
    cuda::memcpy(&result, x + idx, sizeof(f32), CXM_DEVICE_TO_HOST);
    return result;
}