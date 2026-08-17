//
// Created by muham on 14.08.2026.
//

#include "CortexMind/framework/Engine/CUDA/element_wise.cuh"
#include <CortexMind/framework/Engine/CUDA/cast.cuh>
#include <CortexMind/framework/Engine/Kernels/element_wise.cuh>
#include <CortexMind/framework/Engine/Kernels/matrix.cuh>
#include <CortexMind/framework/Engine/Kernels/scalar.cuh>
#include <CortexMind/framework/Engine/Operations/kernel_ops.cuh>
#include <CortexMind/framework/Tools/grid.cuh>

namespace cortex::_fw::nv {
    template<>
    void ElementWise::square<float>(const float* __restrict Xx, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        kernels::element_wise<ops::Square><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::square<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        kernels::element_wise<ops::Square><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::square<tlx::half>(const tlx::half* __restrict Xx, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        kernels::element_wise<ops::Square><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::pow<float>(const float* __restrict Xx, const float value, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Power><<<grid_size, kBlockSize>>>(Xx4, value, Xz4, N);
    }

    template<>
    void ElementWise::pow<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, const tlx::bfloat16 value, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Power><<<grid_size, kBlockSize>>>(Xx4, value, Xz4, N);
    }

    template<>
    void ElementWise::pow<tlx::half>(const tlx::half* __restrict Xx, const tlx::half value, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Power><<<grid_size, kBlockSize>>>(Xx4, value, Xz4, N);
    }

    template<>
    void ElementWise::pow<float>(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Power><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void ElementWise::pow<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, const tlx::bfloat16* __restrict Xy, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Power><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void ElementWise::pow<tlx::half>(const tlx::half* __restrict Xx, const tlx::half* __restrict Xy, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Power><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void ElementWise::sqrt<float>(const float* __restrict Xx, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        kernels::element_wise<ops::Sqrt><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::sqrt<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        kernels::element_wise<ops::Sqrt><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::sqrt<tlx::half>(const tlx::half* __restrict Xx, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        kernels::element_wise<ops::Sqrt><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::rsqrt<float>(const float* __restrict Xx, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        kernels::element_wise<ops::RSqrt><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::rsqrt<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        kernels::element_wise<ops::RSqrt><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::rsqrt<tlx::half>(const tlx::half* __restrict Xx, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        kernels::element_wise<ops::RSqrt><<<grid_size, kBlockSize>>>(
            convert(Xx),
            convert(Xz),
            N
        );
    }

    template<>
    void ElementWise::log<float>(const float* __restrict Xx, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Log><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::log<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Log><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::log<tlx::half>(const tlx::half* __restrict Xx, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Log><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::exp<float>(const float* __restrict Xx, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Exp><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::exp<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Exp><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::exp<tlx::half>(const tlx::half* __restrict Xx, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Exp><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::erf<float>(const float* __restrict Xx, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Erf><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::erf<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Erf><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::erf<tlx::half>(const tlx::half* __restrict Xx, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Erf><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::sin<float>(const float* __restrict Xx, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Sin><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::sin<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Sin><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::sin<tlx::half>(const tlx::half* __restrict Xx, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Sin><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::cos<float>(const float* __restrict Xx, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Cos><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::cos<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Cos><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::cos<tlx::half>(const tlx::half* __restrict Xx, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Cos><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::abs<float>(const float* __restrict Xx, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Abs><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::abs<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Abs><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::abs<tlx::half>(const tlx::half* __restrict Xx, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Abs><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::neg<float>(const float* __restrict Xx, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Neg><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::neg<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Neg><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::neg<tlx::half>(const tlx::half* __restrict Xx, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Neg><<<grid_size, kBlockSize>>>(
            Xx4,
            Xz4,
            N
        );
    }

    template<>
    void ElementWise::rcp<float>(const float* __restrict Xx, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Rcp><<<grid_size, kBlockSize>>>(Xx4, Xz4, N);
    }

    template<>
    void ElementWise::rcp<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Rcp><<<grid_size, kBlockSize>>>(Xx4, Xz4, N);
    }

    template<>
    void ElementWise::rcp<tlx::half>(const tlx::half* __restrict Xx, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::element_wise<ops::Rcp><<<grid_size, kBlockSize>>>(Xx4, Xz4, N);
    }
} //namespace cortex::_fw::nv