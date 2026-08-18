//
// Created by muham on 13.08.2026.
//

#include "CortexMind/framework/Engine/CUDA/scalar.cuh"
#include <CortexMind/framework/Engine/CUDA/cast.cuh>
#include <CortexMind/framework/Engine/Kernels/scalar.cuh>
#include <CortexMind/framework/Engine/Operations/kernel_ops.cuh>
#include <CortexMind/framework/Tools/grid.cuh>
#include <CortexMind/framework/Type/size.hpp>
#include <tlx/concepts.hpp>

using namespace cortex::_fw::nv;
using namespace cortex::_fw;

void ScalarKernel::add(const TensorView &Xx, const float value, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>() {
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::scalar<ops::Addition><<<grid_size, kBlockSize>>>(
            Xx4,
            static_cast<T>(value),
            Xz4,
            Xx.size()
        );
    });
}

void ScalarKernel::sub(const TensorView &Xx, float value, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>() {
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::scalar<ops::Subtraction><<<grid_size, kBlockSize>>>(
            Xx4,
            static_cast<T>(value),
            Xz4,
            Xx.size()
        );
    });
}

void ScalarKernel::mul(const TensorView &Xx, float value, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>() {
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::scalar<ops::Multiplication><<<grid_size, kBlockSize>>>(
            Xx4,
            static_cast<T>(value),
            Xz4,
            Xx.size()
        );
    });
}

void ScalarKernel::div(const TensorView &Xx, float value, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>() {
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::scalar<ops::Division><<<grid_size, kBlockSize>>>(
            Xx4,
            static_cast<T>(value),
            Xz4,
            Xx.size()
        );
    });
}

void ScalarKernel::add(TensorView &Xx, float value) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>() {
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<T*>(Xx.data()));

        kernels::scalar<ops::Addition><<<grid_size, kBlockSize>>>(
            Xx4,
            static_cast<T>(value),
            Xx.size()
        );
    });
}

void ScalarKernel::sub(TensorView &Xx, float value) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>() {
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<T*>(Xx.data()));

        kernels::scalar<ops::Subtraction><<<grid_size, kBlockSize>>>(
            Xx4,
            static_cast<T>(value),
            Xx.size()
        );
    });
}

void ScalarKernel::mul(TensorView &Xx, float value) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>() {
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<T*>(Xx.data()));

        kernels::scalar<ops::Multiplication><<<grid_size, kBlockSize>>>(
            Xx4,
            static_cast<T>(value),
            Xx.size()
        );
    });
}

void ScalarKernel::div(TensorView &Xx, float value) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>() {
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<T*>(Xx.data()));

        kernels::scalar<ops::Division><<<grid_size, kBlockSize>>>(
            Xx4,
            static_cast<T>(value),
            Xx.size()
        );
    });
}