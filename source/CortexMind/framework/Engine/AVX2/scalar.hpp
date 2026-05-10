//
// Created by muham on 8.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_SCALAR_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_SCALAR_HPP

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2-accelerated scalar operations on float arrays.
     *
     * This struct contains optimized functions for common operations
     * (`add`, `sub`, `mul`, `div`) between a float array and a scalar value.
     *
     * All functions use a hybrid approach:
     * - Main loop: 8-wide AVX2 vectorized path
     * - Remainder: Scalar fallback loop
     */
    struct ScalarOp {
        /**
         * @brief Out-of-place addition: `Z[i] = X[i] + value`
         *
         * @param Xx  Input array pointer (read-only)
         * @param value Scalar value to add
         * @param Xz  Output array pointer (can be same as Xx)
         * @param N   Number of elements to process
         */
        static void add(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        /**
         * @brief Out-of-place subtraction: `Z[i] = X[i] - value`
         *
         * @param Xx  Input array pointer (read-only)
         * @param value Scalar value to add
         * @param Xz  Output array pointer (can be same as Xx)
         * @param N   Number of elements to process
         */
        static void sub(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        /**
         * @brief Out-of-place multiplication: `Z[i] = X[i] * value`
         *
         * @param Xx  Input array pointer (read-only)
         * @param value Scalar value to add
         * @param Xz  Output array pointer (can be same as Xx)
         * @param N   Number of elements to process
         */
        static void mul(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        /**
         * @brief Out-of-place division: `Z[i] = X[i] / value`
         *
         * @param Xx  Input array pointer (read-only)
         * @param value Scalar value to add
         * @param Xz  Output array pointer (can be same as Xx)
         * @param N   Number of elements to process
         */
        static void div(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);

        /**
         * @brief In-place addition: `X[i] = X[i] + value`
         *
         * @param Xx  Input/Output array pointer
         * @param value Scalar value to add
         * @param N   Number of elements to process
         */
        static void add(f32* Xx, f32 value, size_t N);
        /**
         * @brief In-place subtraction: `X[i] = X[i] - value`
         *
         * @param Xx  Input/Output array pointer
         * @param value Scalar value to add
         * @param N   Number of elements to process
         */
        static void sub(f32* Xx, f32 value, size_t N);
        /**
         * @brief In-place multiplication: `X[i] = X[i] * value`
         *
         * @param Xx  Input/Output array pointer
         * @param value Scalar value to add
         * @param N   Number of elements to process
         */
        static void mul(f32* Xx, f32 value, size_t N);
        /**
         * @brief In-place division: `X[i] = X[i] / value`
         *
         * @param Xx  Input/Output array pointer
         * @param value Scalar value to add
         * @param N   Number of elements to process
         */
        static void div(f32* Xx, f32 value, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_SCALAR_HPP