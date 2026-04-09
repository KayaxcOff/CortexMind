//
// Created by muham on 9.04.2026.
//

#include "CortexMind/core/Engine/CUDA/reduce.h"
#include <CortexMind/core/Engine/CUDA/Kernels/reduce.cuh>
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/runtime/ctx.h>
#include <cuda_runtime.h>
#include <cmath>

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

f32 ReduceOp::sum(const f32* x, const size_t N) {
    cudaMemset(this->cuda_output, 0, sizeof(f32));

    const size_t shared_mem = (BLOCK_SIZE_1D / WARP_SIZE) * sizeof(f32);
    kernels::reduce_sum<<<grid1d(N), BLOCK_SIZE_1D, shared_mem>>>(x, this->cuda_output, N);

    DeviceSynchronize();
    return *this->host_output;
}

f32 ReduceOp::mean(const f32* x, const size_t N) {
    return this->sum(x, N) / static_cast<f32>(N);
}

f32 ReduceOp::var(const f32* x, const size_t N) {
    const f32 mu = this->mean(x, N);

    cudaMemset(this->cuda_output, 0, sizeof(f32));

    const size_t shared_mem = (BLOCK_SIZE_1D / WARP_SIZE) * sizeof(f32);
    kernels::reduce_var<<<grid1d(N), BLOCK_SIZE_1D, shared_mem>>>(x, mu, this->cuda_output, N);

    DeviceSynchronize();
    return *this->host_output / static_cast<f32>(N);
}

f32 ReduceOp::std(const f32* x, const size_t N) {
    return std::sqrt(this->var(x, N));
}

f32 ReduceOp::min(const f32* x, const size_t N) {
    cublasIsamin(
        runtime::CublasContext::instance().handle,
        static_cast<int>(N),
        x, 1,
        this->cuda_index
    );
    DeviceSynchronize();
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
    DeviceSynchronize();
    const i32 idx = *this->host_index - 1;
    f32 result;
    cuda::memcpy(&result, x + idx, sizeof(f32), CXM_DEVICE_TO_HOST);
    return result;
}

f32 ReduceOp::norm1(const f32* x, const size_t N) {
    cublasSasum(
        runtime::CublasContext::instance().handle,
        static_cast<int>(N),
        x, 1,
        this->cuda_output
    );
    DeviceSynchronize();
    return *this->host_output;
}

f32 ReduceOp::norm2(const f32* x, const size_t N) {
    cublasSnrm2(
        runtime::CublasContext::instance().handle,
        static_cast<int>(N),
        x, 1,
        this->cuda_output
    );
    DeviceSynchronize();
    return *this->host_output;
}

f32 ReduceOp::dot(const f32* Xx, const f32* Xy, const size_t N) {
    cublasSdot(
        runtime::CublasContext::instance().handle,
        static_cast<int>(N),
        Xx, 1,
        Xy, 1,
        this->cuda_output
    );
    DeviceSynchronize();
    return *this->host_output;
}