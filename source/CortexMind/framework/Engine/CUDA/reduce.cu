//
// Created by muham on 9.05.2026.
//

//
// Created by muham on 9.05.2026.
//

#include "CortexMind/framework/Engine/CUDA/reduce.cuh"
#include <CortexMind/framework/Engine/CUDA/Kernels/reduce.cuh>
#include <CortexMind/framework/Engine/CUDA/scalar.cuh>
#include <CortexMind/framework/Engine/CUDA/element_wise.cuh>
#include <CortexMind/framework/Tools/cuda.cuh>
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::cuda;

void ReduceOp::sum(const f32* Xx, f32* Xz, const size_t N) const {
    CXM_CUDA_ASSERT(memset<f32>(Xz, 0, 1));
    constexpr size_t shared_mem = (BLOCK_SIZE_1D / WARP_SIZE) * sizeof(f32);
    kernels::ReduceSum<<<grid1d(N), BLOCK_SIZE_1D, shared_mem>>>(Xx, Xz, N);
}

void ReduceOp::mean(const f32* Xx, f32* Xz, const size_t N) const {
    sum(Xx, Xz, N);

    ScalarKernel::div(Xz, static_cast<f32>(N), 1);
}

void ReduceOp::var(const f32* Xx, f32* Xz, const size_t N) const {
    f32* d_mean;
    f32* d_workspace;

    malloc(&d_mean, sizeof(f32));
    malloc(&d_workspace, N * sizeof(f32));

    mean(Xx, d_mean, N);

    f32 h_mean;
    cudaMemcpy(&h_mean, d_mean, sizeof(f32), cudaMemcpyDeviceToHost);

    ScalarKernel::sub(Xx, h_mean, d_workspace, N);

    ElementWise::square(d_workspace, d_workspace, N);

    this->sum(d_workspace, Xz, N);

    ScalarKernel::div(Xz, static_cast<f32>(N), 1);

    free(d_mean);
    free(d_workspace);
}

void ReduceOp::stdv(const f32* Xx, f32* Xz, const size_t N) const {
    var(Xx, Xz, N);
    ElementWise::sqrt(Xz, Xz, 1);
}

void ReduceOp::argmax(const f32* Xx, i32* Xz, const size_t N) const {
    constexpr size_t shared_mem = (BLOCK_SIZE_1D / WARP_SIZE) * sizeof(kernels::KeyValuePair);
    kernels::ArgMax<<<1, BLOCK_SIZE_1D, shared_mem>>>(Xx, Xz, N);
}

void ReduceOp::argmin(const f32* Xx, i32* Xz, const size_t N) const {
    constexpr size_t shared_mem = (BLOCK_SIZE_1D / WARP_SIZE) * sizeof(kernels::KeyValuePair);
    kernels::ArgMin<<<1, BLOCK_SIZE_1D, shared_mem>>>(Xx, Xz, N);
}


void ReduceOp::sum_dim(const f32* Xx, f32* Xz, size_t outer, size_t dim, size_t inner) const {
    size_t grid_size = outer * inner;
    kernels::ReduceSumDim<BLOCK_SIZE_1D><<<grid_size, BLOCK_SIZE_1D>>>(Xx, Xz, outer, dim, inner);
}

void ReduceOp::mean_dim(const f32* Xx, f32* Xz, size_t outer, size_t dim, size_t inner) const {
    size_t grid_size = outer * inner;
    kernels::ReduceMeanDim<BLOCK_SIZE_1D><<<grid_size, BLOCK_SIZE_1D>>>(Xx, Xz, outer, dim, inner);
}

void ReduceOp::var_dim(const f32* Xx, f32* Xz, const f32* means, size_t outer, size_t dim, size_t inner) const {
    size_t grid_size = outer * inner;
    kernels::ReduceVarDim<BLOCK_SIZE_1D><<<grid_size, BLOCK_SIZE_1D>>>(Xx, Xz, means, outer, dim, inner);
}

void ReduceOp::stdv_dim(const f32* Xx, f32* Xz, const f32* means, size_t outer, size_t dim, size_t inner) const {
    size_t grid_size = outer * inner;
    kernels::ReduceStdvDim<BLOCK_SIZE_1D><<<grid_size, BLOCK_SIZE_1D>>>(Xx, Xz, means, outer, dim, inner);
}

void ReduceOp::min_dim(const f32* Xx, f32* Xz, size_t outer, size_t dim, size_t inner) const {
    size_t grid_size = outer * inner;
    kernels::ReduceMinDim<BLOCK_SIZE_1D><<<grid_size, BLOCK_SIZE_1D>>>(Xx, Xz, outer, dim, inner);
}

void ReduceOp::max_dim(const f32* Xx, f32* Xz, size_t outer, size_t dim, size_t inner) const {
    size_t grid_size = outer * inner;
    kernels::ReduceMaxDim<BLOCK_SIZE_1D><<<grid_size, BLOCK_SIZE_1D>>>(Xx, Xz, outer, dim, inner);
}

void ReduceOp::argmax_dim(const f32* Xx, i32* Xz, size_t outer, size_t dim, size_t inner) const {
    size_t grid_size = outer * inner;
    kernels::ReduceArgMaxDim<BLOCK_SIZE_1D><<<grid_size, BLOCK_SIZE_1D>>>(Xx, Xz, outer, dim, inner);
}

void ReduceOp::argmin_dim(const f32* Xx, i32* Xz, size_t outer, size_t dim, size_t inner) const {
    size_t grid_size = outer * inner;
    kernels::ReduceArgMinDim<BLOCK_SIZE_1D><<<grid_size, BLOCK_SIZE_1D>>>(Xx, Xz, outer, dim, inner);
}

void ReduceOp::norm1_dim(const f32* Xx, f32* Xz, size_t outer, size_t dim, size_t inner) const {
    size_t grid_size = outer * inner;
    kernels::ReduceNorm1Dim<BLOCK_SIZE_1D><<<grid_size, BLOCK_SIZE_1D>>>(Xx, Xz, outer, dim, inner);
}

void ReduceOp::norm2_dim(const f32* Xx, f32* Xz, size_t outer, size_t dim, size_t inner) const {
    size_t grid_size = outer * inner;
    kernels::ReduceNorm2Dim<BLOCK_SIZE_1D><<<grid_size, BLOCK_SIZE_1D>>>(Xx, Xz, outer, dim, inner);
}