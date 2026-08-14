//
// Created by muham on 13.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_SCALAR_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_SCALAR_CUH

#include <CortexMind/framework/Engine/Operations/base.cuh>
#include <CortexMind/framework/Tools/loops.cuh>
#include <CortexMind/runtime/macros.hpp>
#include <tlx/concepts.hpp>
#include <tlx/types.hpp>
#include <cuda_bf16.h>

namespace cortex::_fw::kernels {
    /**
     * @brief Applies a scalar operation element-wise on CUDA.
     *
     * The operation is provided as a template parameter derived from
     * @ref ops::KernelBase. The kernel processes multiple scalar elements
     * through CUDA vector types where possible and handles any remaining
     * elements using a scalar tail loop.
     *
     * @tparam OpType Operation type derived from @ref ops::KernelBase.
     * @param Xx Input array of packed single-precision values.
     * @param value Scalar value supplied to the operation.
     * @param Xz Output array.
     * @param N Number of scalar elements to process.
     */
    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void scalar(const float4* __restrict__ Xx, const float value, float4* __restrict__ Xz, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 4;
        const std::size_t tail_start  = vector_count * 4;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xz[i] = {
                op(Xx[i].x, value), op(Xx[i].y, value),
                op(Xx[i].z, value), op(Xx[i].w, value)
            };
        }

        if (tail_start < N) {
            auto x_s = reinterpret_cast<const float*>(Xx);

            auto z_s = reinterpret_cast<float*>(Xz);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                z_s[i] = op(x_s[i], value);
            }
        }
    }

    /**
     * @brief Applies a scalar operation element-wise to bfloat16 data on CUDA.
     *
     * Processes two bfloat16 values per CUDA thread through
     * @c __nv_bfloat162 and handles an optional remaining element through
     * the scalar tail path.
     *
     * @tparam OpType Operation type derived from @ref ops::KernelBase.
     * @param Xx Input array of packed bfloat16 values.
     * @param value Scalar bfloat16 value supplied to the operation.
     * @param Xz Output array.
     * @param N Number of scalar elements to process.
     */
    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void scalar(const __nv_bfloat162* __restrict__ Xx, const tlx::bfloat16 value, __nv_bfloat162* __restrict__ Xz, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 2;
        const std::size_t tail_start  = vector_count * 2;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xz[i] = {
                op(Xx[i].x, value), op(Xx[i].y, value),
            };
        }

        if (tail_start < N) {
            auto x_s = reinterpret_cast<const tlx::bfloat16*>(Xx);

            auto z_s = reinterpret_cast<tlx::bfloat16*>(Xz);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                z_s[i] = op(x_s[i], value);
            }
        }
    }

    /**
     * @brief Applies a scalar operation element-wise to half-precision data on CUDA.
     *
     * Processes two half-precision values per CUDA thread through
     * @c __nv_half2 and handles any remaining element through the scalar
     * tail path.
     *
     * @tparam OpType Operation type derived from @ref ops::KernelBase.
     * @param Xx Input array of packed half-precision values.
     * @param value Scalar half-precision value supplied to the operation.
     * @param Xz Output array.
     * @param N Number of scalar elements to process.
     */
    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void scalar(const __nv_half2* __restrict__ Xx, const tlx::half value, __nv_half2* __restrict__ Xz, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 2;
        const std::size_t tail_start  = vector_count * 2;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xz[i] = {
                op(Xx[i].x, value), op(Xx[i].y, value),
            };
        }

        if (tail_start < N) {
            auto x_s = reinterpret_cast<const tlx::half*>(Xx);

            auto z_s = reinterpret_cast<tlx::half*>(Xz);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                z_s[i] = op(x_s[i], value);
            }
        }
    }

    /**
     * @brief Applies a scalar operation element-wise in place to float data on CUDA.
     *
     * The input array is also used as the output array. The operation is
     * applied to four float elements at a time through @c float4, followed
     * by a scalar tail path when the element count is not divisible by four.
     *
     * @tparam OpType Operation type derived from @ref ops::KernelBase.
     * @param Xx Input and output array.
     * @param value Scalar value supplied to the operation.
     * @param N Number of scalar elements to process.
     */
    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void scalar_inplace(float4* Xx, const float value, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 4;
        const std::size_t tail_start  = vector_count * 4;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xx[i] = {
                op(Xx[i].x, value), op(Xx[i].y, value),
                op(Xx[i].z, value), op(Xx[i].w, value)
            };
        }

        if (tail_start < N) {
            auto x_s = reinterpret_cast<float*>(Xx);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                x_s[i] = op(x_s[i], value);
            }
        }
    }

    /**
     * @brief Applies a scalar operation element-wise in place to bfloat16 data on CUDA.
     *
     * Processes two bfloat16 values per thread through @c __nv_bfloat162
     * and handles an optional remaining element through the scalar tail path.
     *
     * @tparam OpType Operation type derived from @ref ops::KernelBase.
     * @param Xx Input and output array.
     * @param value Scalar bfloat16 value supplied to the operation.
     * @param N Number of scalar elements to process.
     */
    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void scalar_inplace(__nv_bfloat162* Xx, const tlx::bfloat16 value, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 2;
        const std::size_t tail_start  = vector_count * 2;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xx[i] = {
                op(Xx[i].x, value), op(Xx[i].y, value)
            };
        }

        if (tail_start < N) {
            auto x_s = reinterpret_cast<tlx::bfloat16*>(Xx);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                x_s[i] = op(x_s[i], value);
            }
        }
    }

    /**
     * @brief Applies a scalar operation element-wise in place to half-precision data on CUDA.
     *
     * Processes two half-precision values per thread through @c __nv_half2
     * and handles an optional remaining element through the scalar tail path.
     *
     * @tparam OpType Operation type derived from @ref ops::KernelBase.
     * @param Xx Input and output array.
     * @param value Scalar half-precision value supplied to the operation.
     * @param N Number of scalar elements to process.
     */
    template<tlx::extend<ops::KernelBase> OpType>
    CXM_GLOBAL
    void scalar_inplace(__nv_half2* Xx, const tlx::half value, const std::size_t N) {
        OpType op;

        const std::size_t vector_count = N / 2;
        const std::size_t tail_start  = vector_count * 2;

        CXM_KERNEL_LOOP_1D(i, vector_count) {
            Xx[i] = {
                op(Xx[i].x, value), op(Xx[i].y, value)
            };
        }

        if (tail_start < N) {
            auto x_s = reinterpret_cast<tlx::half*>(Xx);

            CXM_KERNEL_LOOP_TAIL(i, tail_start, N) {
                x_s[i] = op(x_s[i], value);
            }
        }
    }
} //namespace cortex::_fw::kernels

#endif //CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_SCALAR_CUH