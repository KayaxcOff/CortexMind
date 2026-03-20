//
// Created by muham on 13.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_INPLACE_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_INPLACE_HPP

#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief   In-place element-wise arithmetic (X op= ...)
     */
    struct inplace {
        /**
         * @brief   X[i] += Y[i]    for i = 0..idx-1
         * @param   Xx     Array to modify (read + write)
         * @param   Xy     Array to add (read-only)
         * @param   idx    Number of elements
         *
         * @pre     Xx and Xy point to valid, disjoint regions of size ≥ idx
         */
        static void add(f32* Xx, const f32* __restrict Xy, size_t idx);
        /**
         * @brief   X[i] -= Y[i]    for i = 0..idx-1
         * @param   Xx     Array to modify (read + write)
         * @param   Xy     Array to sub (read-only)
         * @param   idx    Number of elements
         *
         * @pre     Xx and Xy point to valid, disjoint regions of size ≥ idx
         */
        static void sub(f32* Xx, const f32* __restrict Xy, size_t idx);
        /**
         * @brief   X[i] *= Y[i]    for i = 0..idx-1
         * @param   Xx     Array to modify (read + write)
         * @param   Xy     Array to mul (read-only)
         * @param   idx    Number of elements
         *
         * @pre     Xx and Xy point to valid, disjoint regions of size ≥ idx
         */
        static void mul(f32* Xx, const f32* __restrict Xy, size_t idx);
        /**
         * @brief   X[i] *= Y[i]    for i = 0..idx-1
         * @param   Xx     Array to modify (read + write)
         * @param   Xy     Array to div (read-only)
         * @param   idx    Number of elements
         *
         * @pre     Xx and Xy point to valid, disjoint regions of size ≥ idx
         */
        static void div(f32* Xx, const f32* __restrict Xy, size_t idx);

        /**
         * @brief   X[i] += value    for i = 0..idx-1
         * @param   Xx     Array to modify
         * @param   value  Scalar to add
         * @param   idx    Number of elements
         */
        static void add(f32* Xx, f32 value, size_t idx);
        /**
         * @brief   X[i] -= value    for i = 0..idx-1
         * @param   Xx     Array to modify
         * @param   value  Scalar to sub
         * @param   idx    Number of elements
         */
        static void sub(f32* Xx, f32 value, size_t idx);
        /**
         * @brief   X[i] *= value    for i = 0..idx-1
         * @param   Xx     Array to modify
         * @param   value  Scalar to mul
         * @param   idx    Number of elements
         */
        static void mul(f32* Xx, f32 value, size_t idx);
        /**
         * @brief   X[i] /= value    for i = 0..idx-1
         * @param   Xx     Array to modify
         * @param   value  Scalar to div
         * @param   idx    Number of elements
         */
        static void div(f32* Xx, f32 value, size_t idx);
    };
} // namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_INPLACE_HPP