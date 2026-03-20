//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_CUH

#include <CortexMind/core/Tools/params.hpp>
#include <cublas_v2.h>

namespace cortex::_fw::cuda {
    /**
     * @brief   High-level wrappers for matrix and element-wise operations on GPU
     *
     * Element-wise functions launch 1D vectorized kernels (float4).
     * matmul launches 2D tiled kernel with shared memory (MAT_TILE × MAT_TILE).
     */
    struct matrix_t {
        /**
         * @brief   Z[i] = X[i] + Y[i]    (out-of-place, GPU)
         * @param   Xx      First input array
         * @param   Xy      Second input array
         * @param   Xz      Output array
         * @param   idx     Number of elements
         */
        static void add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Z[i] = X[i] - Y[i]    (out-of-place, GPU)
         * @param   Xx      First input array
         * @param   Xy      Second input array
         * @param   Xz      Output array
         * @param   idx     Number of elements
         */
        static void sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Z[i] = X[i] * Y[i]    (out-of-place, GPU)
         * @param   Xx      First input array
         * @param   Xy      Second input array
         * @param   Xz      Output array
         * @param   idx     Number of elements
         */
        static void mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Z[i] = X[i] / Y[i]    (out-of-place, GPU)
         * @param   Xx      First input array
         * @param   Xy      Second input array
         * @param   Xz      Output array
         * @param   idx     Number of elements
         */
        static void div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Z[i] = X[i] × Y[i] + W[i]    (cuBLAS GEMM with beta=1)
         * @param   Xx      Multiplicand
         * @param   Xy      Multiplier
         * @param   Xz      Addend
         * @param   Xk      Output
         * @param   idx     Number of elements
         *
         * @note    Uses cuBLAS cublasGemmEx (Tensor Core enabled)
         * @note    Equivalent to GEMM with alpha=1, beta=1
         */
        static void fma(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, f32* __restrict Xk, size_t idx);
        /**
         * @brief   Dense matrix multiplication: Z ← X × Y   (row-major)
         * @param   X       Matrix A (M × K)
         * @param   Y       Matrix B (K × N)
         * @param   Z       Output matrix C (M × N)
         * @param   M       Rows in X and Z
         * @param   K       Columns in X = rows in Y
         * @param   N       Columns in Y and Z
         *
         * @note    Uses cuBLAS cublasGemmEx (Tensor Core enabled)
         * @note    Assumes row-major layout (transposes internally if needed)
         * @note    For very large matrices this is significantly faster than tiled kernel
         */
        static void matmul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, const size_t xIdx, const size_t yIdx, const size_t zIdx);
        static void fill(f32* __restrict Xx, f32 value, size_t idx);
        static void sqrt(const f32* __restrict Xx, f32* __restrict Xz, size_t idx);
        static void pow(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t idx);
    private:
        /**
         * @brief   Returns singleton cuBLAS handle (lazy-initialized)
         * @return  cublasHandle_t (thread-safe, Tensor Core enabled)
         *
         * @note    Created once on first call (static local)
         * @note    Math mode set to CUBLAS_TENSOR_OP_MATH for Ampere+ GPUs
         */
        static cublasHandle_t get_handle();
    };
} // namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_CUH