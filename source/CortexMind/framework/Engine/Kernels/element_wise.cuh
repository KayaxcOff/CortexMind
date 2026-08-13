//
// Created by muham on 13.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_ELEMENT_WISE_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_ELEMENT_WISE_CUH

#include <CortexMind/framework/Engine/Operations/base.cuh>
#include <CortexMind/framework/Tools/loops.cuh>
#include <CortexMind/runtime/macros.hpp>
#include <tlx/concepts.hpp>
#include <tlx/types.hpp>
#include <cuda_bf16.h>

namespace cortex::_fw::kernels {
    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void element_wise(const float4* __restrict__ Xx, float4* __restrict__ Xz, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 4;
        const std::size_t tail_start = vector_count * 4;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xz[i] = {
                op(Xx[i].x), op(Xx[i].y),
                op(Xx[i].z), op(Xx[i].w),
            };
        }

        if (tail_start < N) {
            auto x_s = reinterpret_cast<const float*>(Xx);

            auto z_s = reinterpret_cast<float*>(Xz);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                z_s[i] = op(x_s[i]);
            }
        }
    }

    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void element_wise(const __nv_bfloat162* __restrict__ Xx, __nv_bfloat162* __restrict__ Xz, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 2;
        const std::size_t tail_start = vector_count * 2;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xz[i] = {
                op(Xx[i].x), op(Xx[i].y),
            };
        }

        if (tail_start < N) {
            auto x_s = reinterpret_cast<const tlx::bfloat16*>(Xx);

            auto z_s = reinterpret_cast<tlx::bfloat16*>(Xz);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                z_s[i] = op(x_s[i]);
            }
        }
    }

    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void element_wise(const __nv_half2* __restrict__ Xx, __nv_half2* __restrict__ Xz, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 2;
        const std::size_t tail_start = vector_count * 2;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xz[i] = {
                op(Xx[i].x), op(Xx[i].y),
            };
        }

        if (tail_start < N) {
            auto x_s = reinterpret_cast<const tlx::half*>(Xx);

            auto z_s = reinterpret_cast<tlx::half*>(Xz);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                z_s[i] = op(x_s[i]);
            }
        }
    }
} //namespace cortex::_fw::kernels

#endif //CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_ELEMENT_WISE_CUH