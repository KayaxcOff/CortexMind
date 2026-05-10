//
// Created by muham on 8.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_WISE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_WISE_HPP

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief Element-wise mathematical operations on float arrays (AVX2 optimized).
     *
     * All functions follow the same hybrid pattern:
     * - Early return if `N <= 0`
     * - Main loop: 8-wide AVX2 vectorized path
     * - Remainder: Scalar fallback using `std::` functions
     */
    struct wise {
        /**
         * @brief Element-wise power operation: `Z[i] = X[i] ^ exp`
         *
         * @param Xx  Input array pointer
         * @param exp Exponent value (broadcasted)
         * @param Xz  Output array pointer
         * @param N   Number of elements to process
         *
         * @note Uses `avx2::pow` for vectorized path and `std::pow` for remainder.
         */
        static void pow(const f32* __restrict Xx, f32 exp, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise square root: `Z[i] = sqrt(X[i])`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void sqrt(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise natural logarithm: `Z[i] = log(X[i])`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void log(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise exponential: `Z[i] = e^X[i]`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void exp(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise sine: `Z[i] = sin(X[i])`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void sin(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise absolute value: `Z[i] = |X[i]|`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void abs(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_WISE_HPP