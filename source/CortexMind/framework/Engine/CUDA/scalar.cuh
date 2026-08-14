//
// Created by muham on 13.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_SCALAR_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_SCALAR_CUH

#include <tlx/types.hpp>

namespace cortex::_fw::nv {
    /**
     * @brief Host-side dispatch interface for scalar CUDA operations.
     *
     * Provides scalar arithmetic operations for float, bfloat16, and half
     * precision data. Each operation dispatches the corresponding templated
     * CUDA kernel with the appropriate packed representation and grid size.
     *
     * The class provides both out-of-place operations, which write the result
     * to a separate output buffer, and in-place operations, which modify the
     * input buffer directly.
     */
    struct ScalarKernel {
        /**
         * @brief Adds a scalar value to each element.
         *
         * @param Xx Input array.
         * @param value Scalar value to add.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        static void add(const float* __restrict Xx, float value, float* __restrict Xz, std::size_t N);
        /**
         * @brief Adds a scalar value to each element.
         *
         * @param Xx Input array.
         * @param value Scalar value to add.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        static void add(const tlx::bfloat16* __restrict Xx, tlx::bfloat16 value, tlx::bfloat16* __restrict Xz, std::size_t N);
        /**
         * @brief Adds a scalar value to each element.
         *
         * @param Xx Input array.
         * @param value Scalar value to add.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        static void add(const tlx::half* __restrict Xx, tlx::half value, tlx::half* __restrict Xz, std::size_t N);

        /**
         * @brief Subtracts a scalar value from each element.
         *
         * @param Xx Input array.
         * @param value Scalar value to subtract.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        static void sub(const float* __restrict Xx, float value, float* __restrict Xz, std::size_t N);
        /**
         * @brief Subtracts a scalar value from each element.
         *
         * @param Xx Input array.
         * @param value Scalar value to subtract.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        static void sub(const tlx::bfloat16* __restrict Xx, tlx::bfloat16 value, tlx::bfloat16* __restrict Xz, std::size_t N);
        /**
         * @brief Subtracts a scalar value from each element.
         *
         * @param Xx Input array.
         * @param value Scalar value to subtract.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        static void sub(const tlx::half* __restrict Xx, tlx::half value, tlx::half* __restrict Xz, std::size_t N);

        /**
         * @brief Multiplies each element by a scalar value.
         *
         * @param Xx Input array.
         * @param value Scalar multiplication factor.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        static void mul(const float* __restrict Xx, float value, float* __restrict Xz, std::size_t N);
        /**
         * @brief Multiplies each element by a scalar value.
         *
         * @param Xx Input array.
         * @param value Scalar multiplication factor.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        static void mul(const tlx::bfloat16* __restrict Xx, tlx::bfloat16 value, tlx::bfloat16* __restrict Xz, std::size_t N);
        /**
         * @brief Multiplies each element by a scalar value.
         *
         * @param Xx Input array.
         * @param value Scalar multiplication factor.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        static void mul(const tlx::half* __restrict Xx, tlx::half value, tlx::half* __restrict Xz, std::size_t N);

        /**
         * @brief Divides each element by a scalar value.
         *
         * @param Xx Input array.
         * @param value Scalar divisor.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        static void div(const float* __restrict Xx, float value, float* __restrict Xz, std::size_t N);
        /**
         * @brief Divides each element by a scalar value.
         *
         * @param Xx Input array.
         * @param value Scalar divisor.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        static void div(const tlx::bfloat16* __restrict Xx, tlx::bfloat16 value, tlx::bfloat16* __restrict Xz, std::size_t N);
        /**
         * @brief Divides each element by a scalar value.
         *
         * @param Xx Input array.
         * @param value Scalar divisor.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        static void div(const tlx::half* __restrict Xx, tlx::half value, tlx::half* __restrict Xz, std::size_t N);

        /**
         * @brief Adds a scalar value to each element in place.
         *
         * @param Xx Input and output array.
         * @param value Scalar value to add.
         * @param N Number of elements.
         */
        static void add(float* Xx, float value, std::size_t N);
        /**
         * @brief Adds a scalar value to each element in place.
         *
         * @param Xx Input and output array.
         * @param value Scalar value to add.
         * @param N Number of elements.
         */
        static void add(tlx::bfloat16* Xx, tlx::bfloat16 value, std::size_t N);
        /**
         * @brief Adds a scalar value to each element in place.
         *
         * @param Xx Input and output array.
         * @param value Scalar value to add.
         * @param N Number of elements.
         */
        static void add(tlx::half* Xx, tlx::half value, std::size_t N);


        static void sub(float* Xx, float value, std::size_t N);
        /**
         * @brief Subtracts a scalar value from each element in place.
         *
         * @param Xx Input and output array.
         * @param value Scalar value to subtract.
         * @param N Number of elements.
         */
        static void sub(tlx::bfloat16* Xx, tlx::bfloat16 value, std::size_t N);
        /**
         * @brief Subtracts a scalar value from each element in place.
         *
         * @param Xx Input and output array.
         * @param value Scalar value to subtract.
         * @param N Number of elements.
         */
        static void sub(tlx::half* Xx, tlx::half value, std::size_t N);

        /**
         * @brief Multiplies each element by a scalar value in place.
         *
         * @param Xx Input and output array.
         * @param value Scalar multiplication factor.
         * @param N Number of elements.
         */
        static void mul(float* Xx, float value, std::size_t N);
        /**
         * @brief Multiplies each element by a scalar value in place.
         *
         * @param Xx Input and output array.
         * @param value Scalar multiplication factor.
         * @param N Number of elements.
         */
        static void mul(tlx::bfloat16* Xx, tlx::bfloat16 value, std::size_t N);
        /**
         * @brief Multiplies each element by a scalar value in place.
         *
         * @param Xx Input and output array.
         * @param value Scalar multiplication factor.
         * @param N Number of elements.
         */
        static void mul(tlx::half* Xx, tlx::half value, std::size_t N);

        /**
         * @brief Divides each element by a scalar value in place.
         *
         * @param Xx Input and output array.
         * @param value Scalar divisor.
         * @param N Number of elements.
         */
        static void div(float* Xx, float value, std::size_t N);
        /**
         * @brief Divides each element by a scalar value in place.
         *
         * @param Xx Input and output array.
         * @param value Scalar divisor.
         * @param N Number of elements.
         */
        static void div(tlx::bfloat16* Xx, tlx::bfloat16 value, std::size_t N);
        /**
         * @brief Divides each element by a scalar value in place.
         *
         * @param Xx Input and output array.
         * @param value Scalar divisor.
         * @param N Number of elements.
         */
        static void div(tlx::half* Xx, tlx::half value, std::size_t N);
    };
} //namespace cortex::_fw::nv

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_SCALAR_CUH