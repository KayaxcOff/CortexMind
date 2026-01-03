//
// Created by muham on 29.12.2025.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX_OPS_HPP
#define CORTEXMIND_CORE_ENGINE_AVX_OPS_HPP

#include <core/Engine/AVX/params.hpp>

namespace cortex::_fw::avx2 {

    /// @brief Loads 8 single-precision floats from aligned memory into a 256-bit AVX vector
    /// @param dest Pointer to the aligned float array
    /// @return vec8f containing the loaded values
    inline vec8f load(const float* dest) {
        return _mm256_load_ps(dest);
    }

    /// @brief Stores 8 single-precision floats from a 256-bit AVX vector into aligned memory
    /// @param dest Pointer to the aligned float array
    /// @param src vec8f containing values to store
    inline void store(float* dest, const vec8f& src) {
        _mm256_store_ps(dest, src);
    }

    /// @brief Loads 8 single-precision floats from unaligned memory into a 256-bit AVX vector
    /// @param dest Pointer to the float array (can be unaligned)
    /// @return vec8f containing the loaded values
    inline vec8f load_u(const float* dest) {
        return _mm256_loadu_ps(dest);
    }

    /// @brief Stores 8 single-precision floats from a 256-bit AVX vector into unaligned memory
    /// @param dest Pointer to the float array (can be unaligned)
    /// @param src vec8f containing values to store
    inline void store_u(float* dest, const vec8f& src) {
        _mm256_storeu_ps(dest, src);
    }

    /// @brief Loads a partial number of floats into a 256-bit AVX vector
    /// @param dest Pointer to the source float array
    /// @param idx Number of elements to load (must be <= 8)
    /// @return vec8f vector with the first `idx` elements loaded from `dest`, remaining elements set to zero
    inline vec8f load_partial(const float* dest, const size_t idx) {
        vec8f output = _mm256_setzero_ps();

        for (size_t i = 0; i < idx; ++i) {
            reinterpret_cast<float*>(&output)[i] = dest[i];
        }
        return output;
    }

    /// @brief Stores a partial number of elements from a 256-bit AVX vector into memory
    /// @param dest Pointer to the destination float array
    /// @param src vec8f vector containing the values
    /// @param idx Number of elements to store (must be <= 8)
    inline void store_partial(float* dest, vec8f src, const size_t idx) {
        for (size_t i = 0; i < idx; ++i) {
            dest[i] = reinterpret_cast<float*>(&src)[i];
        }
    }

    /// @brief Adds two 256-bit vectors of single-precision floats element-wise
    /// @param a First input vector
    /// @param b Second input vector
    /// @return vec8f vector containing element-wise sum of `a` and `b`
    inline vec8f add(const vec8f a, const vec8f b) {
        return _mm256_add_ps(a, b);
    }

    /// @brief Subtracts the second 256-bit vector from the first element-wise
    /// @param a First input vector
    /// @param b Second input vector
    /// @return vec8f vector containing element-wise difference of `a - b`
    inline vec8f sub(const vec8f a, const vec8f b) {
        return _mm256_sub_ps(a, b);
    }

    /// @brief Multiplies two 256-bit vectors of single-precision floats element-wise
    /// @param a First input vector
    /// @param b Second input vector
    /// @return vec8f vector containing element-wise product of `a` and `b`
    inline vec8f mul(const vec8f a, const vec8f b) {
        return _mm256_mul_ps(a, b);
    }

    /// @brief Divides the first 256-bit vector by the second element-wise
    /// @param a Dividend vector
    /// @param b Divisor vector
    /// @return vec8f vector containing element-wise division of `a / b`
    inline vec8f div(const vec8f a, const vec8f b) {
        return _mm256_div_ps(a, b);
    }

    /// @brief Broadcasts a single float value to all 8 elements of a 256-bit AVX vector
    /// @param dest The float value to broadcast
    /// @return vec8f vector where all elements are set to `dest`
    inline vec8f broadcast(const float dest) {
        return _mm256_set1_ps(dest);
    }

    /// @brief Creates a 256-bit AVX vector with all elements set to zero
    /// @return vec8f vector where all elements are zero
    inline vec8f zero() {
        return _mm256_setzero_ps();
    }

    /// @brief Computes the square root of each element in a 256-bit AVX vector
    /// @param a Input vector
    /// @return vec8f vector containing element-wise square roots of `a`
    inline vec8f sqrt(const vec8f a) {
        return _mm256_sqrt_ps(a);
    }

    /// @brief Computes the element-wise maximum of two 256-bit AVX vectors
    /// @param a First input vector
    /// @param b Second input vector
    /// @return vec8f vector where each element is the maximum of the corresponding elements in `a` and `b`
    inline vec8f max(const vec8f a, const vec8f b) {
        return _mm256_max_ps(a, b);
    }

    /// @brief Computes the element-wise minimum of two 256-bit AVX vectors
    /// @param a First input vector
    /// @param b Second input vector
    /// @return vec8f vector where each element is the minimum of the corresponding elements in `a` and `b`
    inline vec8f min(const vec8f a, const vec8f b) {
        return _mm256_min_ps(a, b);
    }

    /// @brief Performs element-wise fused multiply-add on three 256-bit AVX vectors
    /// @param a First input vector (multiplier)
    /// @param b Second input vector (multiplicand)
    /// @param c Third input vector (addend)
    /// @return vec8f vector where each element is computed as `a[i] * b[i] + c[i]` with a single fused operation
    inline vec8f fma(const vec8f a, const vec8f b, const vec8f c) {
        return _mm256_fmadd_ps(a, b, c);
    }

    /// @brief Computes the horizontal sum of all 8 elements in a 256-bit AVX vector
    /// @param a Input vec8f vector containing 8 single-precision floats
    /// @return The sum of all elements as a single float
    ///
    /// @details
    /// The 256-bit vector is first split into its lower and upper 128-bit halves.
    /// These halves are added element-wise, resulting in a 128-bit vector containing
    /// the sum of corresponding elements from the lower and upper halves.
    /// Then, two successive horizontal add (_mm_hadd_ps) operations reduce the 4 floats
    /// to a single float, which is returned.
    inline float hadd(const vec8f a) {
        vec4f low = _mm256_castps256_ps128(a);
        const vec4f high = _mm256_extractf128_ps(a, 1);

        low = _mm_add_ps(low, high);
        low = _mm_hadd_ps(low, low);
        low = _mm_hadd_ps(low, low);

        return _mm_cvtss_f32(low);
    }
} // namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX_OPS_HPP