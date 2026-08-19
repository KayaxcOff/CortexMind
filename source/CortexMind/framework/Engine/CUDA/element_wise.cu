//
// Created by muham on 19.08.2026.
//

#include "CortexMind/framework/Engine/CUDA/element_wise.cuh"
#include <CortexMind/framework/Engine/CUDA/cast.cuh>
#include <CortexMind/framework/Engine/Kernels/binary.cuh>
#include <CortexMind/framework/Engine/Kernels/scalar.cuh>
#include <CortexMind/framework/Engine/Kernels/unary.cuh>
#include <CortexMind/framework/Engine/Operations/kernel_ops.cuh>
#include <CortexMind/framework/Tools/grid.cuh>
#include <CortexMind/framework/Type/size.hpp>
#include <tlx/concepts.hpp>

using namespace cortex::_fw::nv;

void ElementWise::square(const TensorView &Xx, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Unary<ops::Square><<<grid_size, kBlockSize>>>(Xx4, Xz4, Xx.size());
    });
}

void ElementWise::pow(const TensorView &Xx, float value, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::scalar<ops::Power><<<grid_size, kBlockSize>>>(Xx4, static_cast<T>(value), Xz4, Xx.size());
    });
}

void ElementWise::pow(const TensorView &Xx, const TensorView &Xy, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xy4 = nv::convert(reinterpret_cast<const T*>(Xy.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Binary<ops::Power><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, Xx.size());
    });
}

void ElementWise::sqrt(const TensorView &Xx, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Unary<ops::Sqrt><<<grid_size, kBlockSize>>>(Xx4, Xz4, Xx.size());
    });
}

void ElementWise::rsqrt(const TensorView &Xx, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Unary<ops::RSqrt><<<grid_size, kBlockSize>>>(Xx4, Xz4, Xx.size());
    });
}

void ElementWise::log(const TensorView &Xx, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Unary<ops::Log><<<grid_size, kBlockSize>>>(Xx4, Xz4, Xx.size());
    });
}

void ElementWise::exp(const TensorView &Xx, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Unary<ops::Exp><<<grid_size, kBlockSize>>>(Xx4, Xz4, Xx.size());
    });
}

void ElementWise::erf(const TensorView &Xx, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Unary<ops::Erf><<<grid_size, kBlockSize>>>(Xx4, Xz4, Xx.size());
    });
}

void ElementWise::sin(const TensorView &Xx, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Unary<ops::Sin><<<grid_size, kBlockSize>>>(Xx4, Xz4, Xx.size());
    });
}

void ElementWise::cos(const TensorView &Xx, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Unary<ops::Cos><<<grid_size, kBlockSize>>>(Xx4, Xz4, Xx.size());
    });
}

void ElementWise::abs(const TensorView &Xx, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Unary<ops::Abs><<<grid_size, kBlockSize>>>(Xx4, Xz4, Xx.size());
    });
}

void ElementWise::neg(const TensorView &Xx, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Unary<ops::Neg><<<grid_size, kBlockSize>>>(Xx4, Xz4, Xx.size());
    });
}

void ElementWise::rcp(const TensorView &Xx, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Unary<ops::Rcp><<<grid_size, kBlockSize>>>(Xx4, Xz4, Xx.size());
    });
}

void ElementWise::inverse(const TensorView &Xx, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Unary<ops::Inverse><<<grid_size, kBlockSize>>>(Xx4, Xz4, Xx.size());
    });
}

void ElementWise::sign(const TensorView &Xx, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Unary<ops::Sign><<<grid_size, kBlockSize>>>(Xx4, Xz4, Xx.size());
    });
}