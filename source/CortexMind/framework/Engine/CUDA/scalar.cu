//
// Created by muham on 13.08.2026.
//

#include "CortexMind/framework/Engine/CUDA/scalar.cuh"
#include <CortexMind/framework/Engine/CUDA/cast.cuh>
#include <CortexMind/framework/Engine/Kernels/scalar.cuh>
#include <CortexMind/framework/Engine/Operations/kernel_ops.cuh>
#include <CortexMind/framework/Tools/grid.cuh>

using namespace cortex::_fw::nv;

void ScalarKernel::add(const float *Xx, const float value, float *Xz, const std::size_t N) {
    const std::int32_t grid_size = grid<4>(N);

    kernels::scalar<ops::Addition><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        convert(Xz),
        N
    );
}

void ScalarKernel::add(const tlx::bfloat16 *Xx, const tlx::bfloat16 value, tlx::bfloat16 *Xz, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar<ops::Addition><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        convert(Xz),
        N
    );
}

void ScalarKernel::add(const tlx::half *Xx, const tlx::half value, tlx::half *Xz, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar<ops::Addition><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        convert(Xz),
        N
    );
}

void ScalarKernel::sub(const float *Xx, const float value, float *Xz, const std::size_t N) {
    const std::int32_t grid_size = grid<4>(N);

    kernels::scalar<ops::Subtraction><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        convert(Xz),
        N
    );
}

void ScalarKernel::sub(const tlx::bfloat16 *Xx, const tlx::bfloat16 value, tlx::bfloat16 *Xz, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar<ops::Subtraction><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        convert(Xz),
        N
    );
}

void ScalarKernel::sub(const tlx::half *Xx, const tlx::half value, tlx::half *Xz, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar<ops::Subtraction><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        convert(Xz),
        N
    );
}

void ScalarKernel::mul(const float *Xx, const float value, float *Xz, const std::size_t N) {
    const std::int32_t grid_size = grid<4>(N);

    kernels::scalar<ops::Multiplication><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        convert(Xz),
        N
    );
}

void ScalarKernel::mul(const tlx::bfloat16 *Xx, const tlx::bfloat16 value, tlx::bfloat16 *Xz, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar<ops::Multiplication><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        convert(Xz),
        N
    );
}

void ScalarKernel::mul(const tlx::half *Xx, const tlx::half value, tlx::half *Xz, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar<ops::Multiplication><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        convert(Xz),
        N
    );
}

void ScalarKernel::div(const float *Xx, const float value, float *Xz, const std::size_t N) {
    const std::int32_t grid_size = grid<4>(N);

    kernels::scalar<ops::Division><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        convert(Xz),
        N
    );
}

void ScalarKernel::div(const tlx::bfloat16 *Xx, const tlx::bfloat16 value, tlx::bfloat16 *Xz, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar<ops::Division><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        convert(Xz),
        N
    );
}

void ScalarKernel::div(const tlx::half *Xx, const tlx::half value, tlx::half *Xz, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar<ops::Division><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        convert(Xz),
        N
    );
}

void ScalarKernel::add(float *Xx, const float value, const std::size_t N) {
    const std::int32_t grid_size = grid<4>(N);

    kernels::scalar_inplace<ops::Addition><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        N
    );
}

void ScalarKernel::add(tlx::bfloat16 *Xx, const tlx::bfloat16 value, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar_inplace<ops::Addition><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        N
    );
}

void ScalarKernel::add(tlx::half *Xx, const tlx::half value, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar_inplace<ops::Addition><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        N
    );
}

void ScalarKernel::sub(float *Xx, const float value, const std::size_t N) {
    const std::int32_t grid_size = grid<4>(N);

    kernels::scalar_inplace<ops::Subtraction><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        N
    );
}

void ScalarKernel::sub(tlx::bfloat16 *Xx, const tlx::bfloat16 value, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar_inplace<ops::Subtraction><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        N
    );
}

void ScalarKernel::sub(tlx::half *Xx, const tlx::half value, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar_inplace<ops::Subtraction><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        N
    );
}

void ScalarKernel::mul(float *Xx, const float value, const std::size_t N) {
    const std::int32_t grid_size = grid<4>(N);

    kernels::scalar_inplace<ops::Multiplication><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        N
    );
}

void ScalarKernel::mul(tlx::bfloat16 *Xx, const tlx::bfloat16 value, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar_inplace<ops::Multiplication><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        N
    );
}

void ScalarKernel::mul(tlx::half *Xx, const tlx::half value, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar_inplace<ops::Multiplication><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        N
    );
}

void ScalarKernel::div(float *Xx, const float value, const std::size_t N) {
    const std::int32_t grid_size = grid<4>(N);

    kernels::scalar_inplace<ops::Division><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        N
    );
}

void ScalarKernel::div(tlx::bfloat16 *Xx, const tlx::bfloat16 value, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar_inplace<ops::Division><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        N
    );
}

void ScalarKernel::div(tlx::half *Xx, const tlx::half value, const std::size_t N) {
    const std::int32_t grid_size = grid<2>(N);

    kernels::scalar_inplace<ops::Division><<<grid_size, kBlockSize>>>(
        convert(Xx),
        value,
        N
    );
}