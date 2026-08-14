//
// Created by muham on 13.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_SCALAR_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_SCALAR_CUH

#include <tlx/concepts.hpp>

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
         * @brief Adds a scalar value to every element of an input array.
         *
         * Computes:
         * @code
         * Xz[i] = Xx[i] + value
         * @endcode
         *
         * @tparam T Floating-point-like element type.
         * @param Xx Input array.
         * @param value Scalar value to add.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        template<tlx::float_like T>
        static void add(const T* __restrict Xx, T value, T* __restrict Xz, std::size_t N);
        /**
         * @brief Subtracts a scalar value from every element of an input array.
         *
         * Computes:
         * @code
         * Xz[i] = Xx[i] - value
         * @endcode
         *
         * @tparam T Floating-point-like element type.
         * @param Xx Input array.
         * @param value Scalar value to subtract.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        template<tlx::float_like T>
        static void sub(const T* __restrict Xx, T value, T* __restrict Xz, std::size_t N);
        /**
         * @brief Multiplies every element of an input array by a scalar value.
         *
         * Computes:
         * @code
         * Xz[i] = Xx[i] * value
         * @endcode
         *
         * @tparam T Floating-point-like element type.
         * @param Xx Input array.
         * @param value Scalar multiplication factor.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        template<tlx::float_like T>
        static void mul(const T* __restrict Xx, T value, T* __restrict Xz, std::size_t N);
        /**
         * @brief Divides every element of an input array by a scalar value.
         *
         * Computes:
         * @code
         * Xz[i] = Xx[i] / value
         * @endcode
         *
         * @tparam T Floating-point-like element type.
         * @param Xx Input array.
         * @param value Scalar divisor.
         * @param Xz Output array.
         * @param N Number of elements.
         */
        template<tlx::float_like T>
        static void div(const T* __restrict Xx, T value, T* __restrict Xz, std::size_t N);

        /**
         * @brief Adds a scalar value to every element in place.
         *
         * Computes:
         * @code
         * Xx[i] += value
         * @endcode
         *
         * @tparam T Floating-point-like element type.
         * @param Xx Input and output array.
         * @param value Scalar value to add.
         * @param N Number of elements.
         */
        template<tlx::float_like T>
        static void add(T* Xx, T value, std::size_t N);
        /**
         * @brief Subtracts a scalar value from every element in place.
         *
         * Computes:
         * @code
         * Xx[i] -= value
         * @endcode
         *
         * @tparam T Floating-point-like element type.
         * @param Xx Input and output array.
         * @param value Scalar value to subtract.
         * @param N Number of elements.
         */
        template<tlx::float_like T>
        static void sub(T* Xx, T value, std::size_t N);
        /**
         * @brief Multiplies every element by a scalar value in place.
         *
         * Computes:
         * @code
         * Xx[i] *= value
         * @endcode
         *
         * @tparam T Floating-point-like element type.
         * @param Xx Input and output array.
         * @param value Scalar multiplication factor.
         * @param N Number of elements.
         */
        template<tlx::float_like T>
        static void mul(T* Xx, T value, std::size_t N);
        /**
         * @brief Divides every element by a scalar value in place.
         *
         * Computes:
         * @code
         * Xx[i] /= value
         * @endcode
         *
         * @tparam T Floating-point-like element type.
         * @param Xx Input and output array.
         * @param value Scalar divisor.
         * @param N Number of elements.
         */
        template<tlx::float_like T>
        static void div(T* Xx, T value, std::size_t N);
    };
} //namespace cortex::_fw::nv

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_SCALAR_CUH