//
// Created by muham on 12.04.2026.
//

#include "CortexMind/core/Engine/CUDA/softmax.h"
#include <CortexMind/core/Engine/CUDA/Kernels/softmax.cuh>
#include <CortexMind/core/Tools/utils.cuh>
#include <cfloat>

using namespace cortex::_fw::cuda;

Softmax::Softmax() {
    CXM_CUDA_ASSERT(
        host::allocate(reinterpret_cast<void**>(&this->host_buf),
                       sizeof(f32) * 2, CXM_HOST_ALLOC_MAPPED),
        "cortex::_fw::cuda::Softmax::Softmax()"
    );
    CXM_CUDA_ASSERT(
        map(reinterpret_cast<void**>(&this->cuda_buf), this->host_buf),
        "cortex::_fw::cuda::Softmax::Softmax()"
    );
}

Softmax::~Softmax() {
    CXM_CUDA_ASSERT(host::free(this->host_buf), "cortex::_fw::cuda::Softmax::~Softmax()");
}

void Softmax::forward(f32* Xx, const size_t N) {
    const size_t shared_mem = (BLOCK_SIZE_1D / WARP_SIZE) * sizeof(f32);

    f32* cuda_max = this->cuda_buf;
    f32* cuda_sum = this->cuda_buf + 1;

    this->host_buf[0] = -FLT_MAX;
    this->host_buf[1] = 0.0f;

    kernels::softmax_max<<<1, BLOCK_SIZE_1D, shared_mem>>>(Xx, cuda_max, N);
    DeviceSynchronize();

    const f32 x_max = this->host_buf[0];

    kernels::softmax_exp_sum<<<grid1d(N), BLOCK_SIZE_1D, shared_mem>>>(Xx, x_max, cuda_sum, N);
    DeviceSynchronize();

    const f32 x_sum = this->host_buf[1];

    kernels::softmax_normalize<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, x_sum, N);
}