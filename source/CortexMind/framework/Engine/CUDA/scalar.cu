//
// Created by muham on 13.08.2026.
//

#include "CortexMind/framework/Engine/CUDA/scalar.cuh"
#include <CortexMind/framework/Engine/CUDA/cast.cuh>
#include <CortexMind/framework/Engine/Kernels/scalar.cuh>
#include <CortexMind/framework/Engine/Operations/kernel_ops.cuh>
#include <CortexMind/framework/Tools/grid.cuh>

namespace cortex::_fw::nv {
    template<>
    void ScalarKernel::add<float>(const float* __restrict Xx, const float value, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Addition><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            Xz4,
            N
        );
    }

    template<>
    void ScalarKernel::add<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, const tlx::bfloat16 value, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Addition><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            Xz4,
            N
        );
    }

    template<>
    void ScalarKernel::add<tlx::half>(const tlx::half* __restrict Xx, const tlx::half value, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Addition><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            Xz4,
            N
        );
    }

    template<>
    void ScalarKernel::sub<float>(const float* __restrict Xx, const float value, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Subtraction><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            Xz4,
            N
        );
    }

    template<>
    void ScalarKernel::sub<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, const tlx::bfloat16 value, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Subtraction><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            Xz4,
            N
        );
    }

    template<>
    void ScalarKernel::sub<tlx::half>(const tlx::half* __restrict Xx, const tlx::half value, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Subtraction><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            Xz4,
            N
        );
    }

    template<>
    void ScalarKernel::mul<float>(const float* __restrict Xx, const float value, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Multiplication><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            Xz4,
            N
        );
    }

    template<>
    void ScalarKernel::mul<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, const tlx::bfloat16 value, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Multiplication><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            Xz4,
            N
        );
    }

    template<>
    void ScalarKernel::mul<tlx::half>(const tlx::half* __restrict Xx, const tlx::half value, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Multiplication><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            Xz4,
            N
        );
    }

    template<>
    void ScalarKernel::div<float>(const float* __restrict Xx, const float value, float* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Division><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            Xz4,
            N
        );
    }

    template<>
    void ScalarKernel::div<tlx::bfloat16>(const tlx::bfloat16* __restrict Xx, const tlx::bfloat16 value, tlx::bfloat16* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Division><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            Xz4,
            N
        );
    }

    template<>
    void ScalarKernel::div<tlx::half>(const tlx::half* __restrict Xx, const tlx::half value, tlx::half* __restrict Xz, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);
        const auto Xz4 = convert(Xz);

        kernels::scalar<ops::Division><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            Xz4,
            N
        );
    }

    template<>
    void ScalarKernel::add<float>(float* Xx, const float value, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);

        kernels::scalar<ops::Addition><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            N
        );
    }

    template<>
    void ScalarKernel::add<tlx::bfloat16>(tlx::bfloat16* Xx, const tlx::bfloat16 value, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);

        kernels::scalar<ops::Addition><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            N
        );
    }

    template<>
    void ScalarKernel::add<tlx::half>(tlx::half* Xx, const tlx::half value, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);

        kernels::scalar<ops::Addition><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            N
        );
    }

    template<>
    void ScalarKernel::sub<float>(float* Xx, const float value, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);

        kernels::scalar<ops::Subtraction><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            N
        );
    }

    template<>
    void ScalarKernel::sub<tlx::bfloat16>(tlx::bfloat16* Xx, const tlx::bfloat16 value, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);

        kernels::scalar<ops::Subtraction><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            N
        );
    }

    template<>
    void ScalarKernel::sub<tlx::half>(tlx::half* Xx, const tlx::half value, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);

        kernels::scalar<ops::Subtraction><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            N
        );
    }

    template<>
    void ScalarKernel::mul<float>(float* Xx, const float value, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);

        kernels::scalar<ops::Multiplication><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            N
        );
    }

    template<>
    void ScalarKernel::mul<tlx::bfloat16>(tlx::bfloat16* Xx, const tlx::bfloat16 value, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);

        kernels::scalar<ops::Multiplication><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            N
        );
    }

    template<>
    void ScalarKernel::mul<tlx::half>(tlx::half* Xx, const tlx::half value, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);

        kernels::scalar<ops::Multiplication><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            N
        );
    }

    template<>
    void ScalarKernel::div<float>(float* Xx, const float value, const std::size_t N) {
        const int grid_size = grid<4>(N);

        const auto Xx4 = convert(Xx);

        kernels::scalar<ops::Division><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            N
        );
    }

    template<>
    void ScalarKernel::div<tlx::bfloat16>(tlx::bfloat16* Xx, const tlx::bfloat16 value, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);

        kernels::scalar<ops::Division><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            N
        );
    }

    template<>
    void ScalarKernel::div<tlx::half>(tlx::half* Xx, const tlx::half value, const std::size_t N) {
        const int grid_size = grid<2>(N);

        const auto Xx4 = convert(Xx);

        kernels::scalar<ops::Division><<<grid_size, kBlockSize>>>(
            Xx4,
            value,
            N
        );
    }
} //namespace cortex::_fw::nv