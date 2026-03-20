//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_INPLACE_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_INPLACE_CUH

#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief   In-place vector-vector element-wise operations on GPU
     *
     * Each method applies the corresponding binary operation element-wise
     * and modifies the first input array directly (in-place).
     *
     * Typical usage:
     * @code
     *     cuda::inplace::mul(d_weights, d_gradients, n_elements);
     *     cudaDeviceSynchronize();  // if immediate result needed
     * @endcode
     */
    struct inplace {
        /**
         * @brief   X[i] += Y[i]    (in-place)
         * @param   Xx      Array to modify (device pointer)
         * @param   Xy      Array to add (device pointer)
         * @param   idx     Number of elements
         */
        static void add(f32* __restrict Xx, const f32* __restrict Xy, size_t idx);
        /**
         * @brief   X[i] -= Y[i]    (in-place)
         * @param   Xx      Array to modify (device pointer)
         * @param   Xy      Array to add (device pointer)
         * @param   idx     Number of elements
         */
        static void sub(f32* __restrict Xx, const f32* __restrict Xy, size_t idx);
        /**
         * @brief   X[i] *= Y[i]    (in-place)
         * @param   Xx      Array to modify (device pointer)
         * @param   Xy      Array to add (device pointer)
         * @param   idx     Number of elements
         */
        static void mul(f32* __restrict Xx, const f32* __restrict Xy, size_t idx);
        /**
         * @brief   X[i] /= Y[i]    (in-place)
         * @warning If any Y[i] == 0 → Inf or NaN in corresponding X[i]
         * @param   Xx      Array to modify (device pointer)
         * @param   Xy      Array to add (device pointer)
         * @param   idx     Number of elements
         */
        static void div(f32* __restrict Xx, const f32* __restrict Xy, size_t idx);
    };
    /**
     * @brief   In-place scalar broadcast operations on GPU
     *
     * Each method applies the scalar operation element-wise and modifies
     * the input array directly (in-place).
     *
     * Typical usage:
     * @code
     *     cuda::inplace_scalar::mul(d_data, 1.5f, n_elements);
     * @endcode
     */
    struct inplace_scalar {
        /**
         * @brief   X[i] += value
         * @param   Xx      Array to modify
         * @param   value   Scalar addend
         * @param   idx     Number of elements
         */
        static void add(f32* __restrict Xx, f32 value, size_t idx);
        /**
         * @brief   X[i] -= value
         * @param   Xx      Array to modify
         * @param   value   Scalar addend
         * @param   idx     Number of elements
         */
        static void sub(f32* __restrict Xx, f32 value, size_t idx);
        /**
         * @brief   X[i] *= value
         * @param   Xx      Array to modify
         * @param   value   Scalar addend
         * @param   idx     Number of elements
         */
        static void mul(f32* __restrict Xx, f32 value, size_t idx);
        /**
         * @brief   X[i] /= value
         * @param   Xx      Array to modify
         * @param   value   Scalar addend
         * @param   idx     Number of elements
         */
        static void div(f32* __restrict Xx, f32 value, size_t idx);
    };
} // namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_INPLACE_CUH