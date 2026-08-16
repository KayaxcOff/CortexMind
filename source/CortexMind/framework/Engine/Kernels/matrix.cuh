//
// Created by muham on 16.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_MATRIX_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_MATRIX_CUH

#include <CortexMind/framework/Engine/Operations/base.cuh>
#include <CortexMind/framework/Tools/loops.cuh>
#include <tlx/concepts.hpp>
#include <tlx/types.hpp>
#include <cuda_bf16.h>

namespace cortex::_fw::kernels {
    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void BinaryKernel(const float4* __restrict Xx, const float4* __restrict Xy, float4* __restrict Xz, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 4;
        const std::size_t tail_start  = vector_count * 4;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xz[i] = {
                op(Xx[i].x, Xy[i].x), op(Xx[i].y, Xy[i].y),
                op(Xx[i].z, Xy[i].z), op(Xx[i].w, Xy[i].w),
            };
        }

        if (tail_start < N) {
            auto x = reinterpret_cast<const float*>(Xx);
            auto y = reinterpret_cast<const float*>(Xy);
            auto z = reinterpret_cast<float*>(Xz);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                z[i] = op(x[i], y[i]);
            }
        }
    }

    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void BinaryKernel(const __nv_bfloat162* __restrict Xx, const __nv_bfloat162* __restrict Xy, __nv_bfloat162* __restrict Xz, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 2;
        const std::size_t tail_start  = vector_count * 2;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xz[i] = {
                op(Xx[i].x, Xy[i].x), op(Xx[i].y, Xy[i].y),
            };
        }

        if (tail_start < N) {
            auto x = reinterpret_cast<const tlx::bfloat16*>(Xx);
            auto y = reinterpret_cast<const tlx::bfloat16*>(Xy);
            auto z = reinterpret_cast<tlx::bfloat16*>(Xz);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                z[i] = op(x[i], y[i]);
            }
        }
    }

    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void BinaryKernel(const __nv_half2* __restrict Xx, const __nv_half2* __restrict Xy, __nv_half2* __restrict Xz, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 2;
        const std::size_t tail_start  = vector_count * 2;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xz[i] = {
                op(Xx[i].x, Xy[i].x), op(Xx[i].y, Xy[i].y),
            };
        }

        if (tail_start < N) {
            auto x = reinterpret_cast<const tlx::half*>(Xx);
            auto y = reinterpret_cast<const tlx::half*>(Xy);
            auto z = reinterpret_cast<tlx::half*>(Xz);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                z[i] = op(x[i], y[i]);
            }
        }
    }

    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void BinaryKernel(float4* Xx, const float4* __restrict Xy, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 4;
        const std::size_t tail_start  = vector_count * 4;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xx[i] = {
                op(Xx[i].x, Xy[i].x), op(Xx[i].y, Xy[i].y),
                op(Xx[i].z, Xy[i].z), op(Xx[i].w, Xy[i].w),
            };
        }

        if (tail_start < N) {
            auto x = reinterpret_cast<float*>(Xx);
            auto y = reinterpret_cast<const float*>(Xy);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                x[i] = op(x[i], y[i]);
            }
        }
    }

    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void BinaryKernel(__nv_bfloat162* Xx, const __nv_bfloat162* __restrict Xy, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 2;
        const std::size_t tail_start  = vector_count * 2;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xx[i] = {
                op(Xx[i].x, Xy[i].x), op(Xx[i].y, Xy[i].y)
            };
        }

        if (tail_start < N) {
            auto x = reinterpret_cast<tlx::bfloat16*>(Xx);
            auto y = reinterpret_cast<const tlx::bfloat16*>(Xy);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                x[i] = op(x[i], y[i]);
            }
        }
    }

    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void BinaryKernel(__nv_half2* Xx, const __nv_half2* __restrict Xy, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 2;
        const std::size_t tail_start  = vector_count * 2;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xx[i] = {
                op(Xx[i].x, Xy[i].x), op(Xx[i].y, Xy[i].y)
            };
        }

        if (tail_start < N) {
            auto x = reinterpret_cast<tlx::half*>(Xx);
            auto y = reinterpret_cast<const tlx::half*>(Xy);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                x[i] = op(x[i], y[i]);
            }
        }
    }
} //namespace cortex::_fw::kernels

#endif //CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_MATRIX_CUH