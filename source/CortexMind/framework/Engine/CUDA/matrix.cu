//
// Created by muham on 17.08.2026.
//

#include "CortexMind/framework/Engine/CUDA/matrix.cuh"
#include <CortexMind/framework/Engine/CUDA/cast.cuh>
#include <CortexMind/framework/Engine/Kernels/matrix.cuh>
#include <CortexMind/framework/Engine/Operations/kernel_ops.cuh>
#include <CortexMind/framework/Tools/grid.cuh>

namespace cortex::_fw::nv {
    template<>
    void Matrix::add<float>(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Addition><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::add<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, const tlx::bfloat16* __restrict Xy, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Addition><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::add<tlx::half>(const tlx::half* __restrict Xx, const tlx::half* __restrict Xy, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Addition><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::sub<float>(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Subtraction><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::sub<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, const tlx::bfloat16* __restrict Xy, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Subtraction><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::sub<tlx::half>(const tlx::half* __restrict Xx, const tlx::half* __restrict Xy, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Subtraction><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::mul<float>(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Multiplication><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::mul<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, const tlx::bfloat16* __restrict Xy, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Multiplication><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::mul<tlx::half>(const tlx::half* __restrict Xx, const tlx::half* __restrict Xy, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Multiplication><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::div<float>(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Division><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::div<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, const tlx::bfloat16* __restrict Xy, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Division><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::div<tlx::half>(const tlx::half* __restrict Xx, const tlx::half* __restrict Xy, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Division><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::max<float>(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Max><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::max<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, const tlx::bfloat16* __restrict Xy, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Max><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::max<tlx::half>(const tlx::half* __restrict Xx, const tlx::half* __restrict Xy, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Max><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::min<float>(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Min><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::min<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, const tlx::bfloat16* __restrict Xy, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Min><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }

    template<>
    void Matrix::min<tlx::half>(const tlx::half* __restrict Xx, const tlx::half* __restrict Xy, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xy4 = convert(Xy);
        const auto Xz4 = convert(Xz);

        kernels::BinaryKernel<ops::Min><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, N);
    }
} //namespace cortex::_fw::nv