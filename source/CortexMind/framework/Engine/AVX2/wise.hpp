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
         * @brief Element-wise sqrt root: `Z[i] = sqrt(X[i])`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void sqrt(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise rsqrt root: `Z[i] = 1 / sqrt(X[i])`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void rsqrt(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise natural logarithm: `Z[i] = log(X[i])`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void log(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise natural logarithm: `Z[i] = log2(X[i])`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void log2(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise natural logarithm: `Z[i] = log10(X[i])`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void log10(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise exponential: `Z[i] = e^X[i]`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void exp(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise exponential: `Z[i] = 2^X[i]`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void exp2(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise exponential: `Z[i] = 10^X[i]`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void exp10(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise error function: `Z[i] = erf(X[i])`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void erf(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise sine: `Z[i] = sin(X[i])`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void sin(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise cose: `Z[i] = cos(X[i])`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void cos(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise tangent: `Z[i] = tan(X[i])`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void tan(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise cotangent: `Z[i] = 1 / tan(X[i])`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void cot(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise absolute value: `Z[i] = |X[i]|`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void abs(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise neg value: `Z[i] = -X[i]`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void neg(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise neg value: `Z[i] < 0 = -1 | Z[i] = 0 = 0 | Z[i] > o = 1`
         *
         * @param Xx Input array pointer
         * @param Xz Output array pointer
         * @param N  Number of elements
         */
        static void sign(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise clamp operation: `Z[i] = clamp(X[i], min_val, max_val)`
         *
         * @param Xx      Input array
         * @param min_val Lower bound
         * @param max_val Upper bound
         * @param Xz      Output array
         * @param N       Number of elements
         */
        static void clamp(const f32* Xx, f32 min_val, f32 max_val, f32* Xz, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_WISE_HPP