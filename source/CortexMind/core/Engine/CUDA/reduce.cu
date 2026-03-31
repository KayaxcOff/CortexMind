//
// Created by muham on 31.03.2026.
//

#include "CortexMind/core/Engine/CUDA/reduce.h"
#include <CortexMind/core/Kernels/reduce.cuh>
#include <CortexMind/core/Tools/utilities.hpp>
#include <CortexMind/runtime/CUDA/context.cuh>
#include <CortexMind/framework/Memory/transform.cuh>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw::sys;
using namespace cortex::runtime;
using namespace cortex::_fw;

namespace {
    template<typename KernelFn>
    f32 launch(KernelFn fn, f32 init = 0.0f) {
        auto& ctx = CudaContext::get();


        transform<f32>::upload(init, ctx.d_result, ctx.stream);

        // kernel
        fn(ctx.d_result, ctx.stream);

        // stream sync — async transfer öncesi tamamlanmalı
        cudaStreamSynchronize(ctx.stream);

        // GPU → CPU
        f32 result;
        transform<f32>::download(ctx.d_result, result, ctx.stream);
        cudaStreamSynchronize(ctx.stream);

        return result;
    }
}

f32 Reduce::sum(const bf16* Xx, const size_t N) {
    if (N == 0) return 0.0f;
    return launch([&](f32* out, cudaStream_t stream) {
        kernels::reduce_sum_kernel<<<grid1d(N), BLOCK_SIZE_1D, 0, stream>>>(Xx, out, N);
    });
}

f32 Reduce::mean(const bf16* Xx, const size_t N) {
    if (N == 0) return 0.0f;
    return sum(Xx, N) / static_cast<f32>(N);
}

f32 Reduce::variance(const bf16* Xx, const size_t N, const bool unbiased) {
    if (N < 2) return 0.0f;

    // İki aşama: önce mean, sonra sum of squared diffs
    const f32 mu = mean(Xx, N);

    const f32 sum_sq = launch([&](f32* out, cudaStream_t stream) {
        kernels::reduce_variance_kernel<<<grid1d(N), BLOCK_SIZE_1D, 0, stream>>>(
            Xx, mu, out, N);
    });

    return sum_sq / static_cast<f32>(unbiased ? N - 1 : N);
}

f32 Reduce::max(const bf16* Xx, const size_t N) {
    if (N == 0) return 0.0f;
    return launch([&](f32* out, cudaStream_t stream) {
        kernels::reduce_max_kernel<<<grid1d(N), BLOCK_SIZE_1D, 0, stream>>>(Xx, out, N);
    }, -CXM_F32_MAX);
}

f32 Reduce::min(const bf16* Xx, const size_t N) {
    if (N == 0) return 0.0f;
    return launch([&](f32* out, cudaStream_t stream) {
        kernels::reduce_min_kernel<<<grid1d(N), BLOCK_SIZE_1D, 0, stream>>>(Xx, out, N);
    }, CXM_F32_MAX);
}