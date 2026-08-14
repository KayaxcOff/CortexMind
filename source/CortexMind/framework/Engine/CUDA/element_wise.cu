//
// Created by muham on 14.08.2026.
//

#include "CortexMind/framework/Engine/CUDA/element_wise.cuh"
#include <CortexMind/framework/Engine/CUDA/cast.cuh>
#include <CortexMind/framework/Engine/Kernels/element_wise.cuh>
#include <CortexMind/framework/Engine/Operations/kernel_ops.cuh>
#include <CortexMind/framework/Tools/grid.cuh>

namespace cortex::_fw::nv {
    template<>
    void ElementWise::square<float>(const float* __restrict Xx, float* __restrict Xz, const std::size_t N) {
        const std::int32_t grid_size = grid<4>(N);

        kernels::element_wise<ops::Square><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::square<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const std::int32_t grid_size = grid<2>(N);

        kernels::element_wise<ops::Square><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::square<tlx::half>(const tlx::half* __restrict Xx, tlx::half* __restrict Xz, const std::size_t N) {
        const std::int32_t grid_size = grid<2>(N);

        kernels::element_wise<ops::Square><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::sqrt<float>(const float* __restrict Xx, float* __restrict Xz, const std::size_t N) {
        const std::int32_t grid_size = grid<4>(N);

        kernels::element_wise<ops::Sqrt><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::sqrt<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const std::int32_t grid_size = grid<2>(N);

        kernels::element_wise<ops::Sqrt><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::sqrt<tlx::half>(const tlx::half* __restrict Xx, tlx::half* __restrict Xz, const std::size_t N) {
        const std::int32_t grid_size = grid<2>(N);

        kernels::element_wise<ops::Sqrt><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::rsqrt<float>(const float* __restrict Xx, float* __restrict Xz, std::size_t N) {
        const std::int32_t grid_size = grid<4>(N);

        kernels::element_wise<ops::RSqrt><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::rsqrt<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const std::int32_t grid_size = grid<2>(N);

        kernels::element_wise<ops::RSqrt><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::rsqrt<tlx::half>(const tlx::half* __restrict Xx, tlx::half* __restrict Xz, const std::size_t N) {
        const std::int32_t grid_size = grid<2>(N);

        kernels::element_wise<ops::RSqrt><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }
} //namespace cortex::_fw::nv