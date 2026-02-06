//
// Created by muham on 2.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX_FUNCS_HPP
#define CORTEXMIND_CORE_ENGINE_AVX_FUNCS_HPP

#include <CortexMind/core/Engine/AVX/params.hpp>
#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::avx2 {

    /**
     * @brief Load 8 aligned single-precision floats into a SIMD register.
     *
     * @param dest Pointer to 32-byte aligned memory.
     * @return SIMD vector containing loaded values.
     *
     * @note The memory address must be 32-byte aligned.
     */
    inline vec8f load(const f32* dest) { return _mm256_load_ps(dest); }
    /**
     * @brief Store a SIMD vector into aligned memory.
     *
     * @param dest Pointer to 32-byte aligned memory.
     * @param src SIMD vector to store.
     */
    inline void store(f32* dest, const vec8f src) { _mm256_store_ps(dest, src); }

    /**
     * @brief Load 8 unaligned single-precision floats into a SIMD register.
     *
     * @param dest Pointer to memory (alignment not required).
     * @return SIMD vector containing loaded values.
     */
    inline vec8f loadu(const f32* dest) { return _mm256_loadu_ps(dest); }

    /**
     * @brief Store a SIMD vector into unaligned memory.
     *
     * @param dest Pointer to memory (alignment not required).
     * @param src SIMD vector to store.
     */
     inline void storeu(f32* dest, const vec8f src) { _mm256_storeu_ps(dest, src); }

    /**
     * @brief Element-wise addition of two SIMD vectors.
     */
    inline vec8f add(const vec8f _x, const vec8f _y) { return _mm256_add_ps(_x, _y); }

    /**
     * @brief Element-wise subtraction of two SIMD vectors.
     */
    inline vec8f sub(const vec8f _x, const vec8f _y) { return _mm256_sub_ps(_x, _y); }

    /**
     * @brief Element-wise multiplication of two SIMD vectors.
     */
    inline vec8f mul(const vec8f _x, const vec8f _y) { return _mm256_mul_ps(_x, _y); }

    /**
     * @brief Element-wise division of two SIMD vectors.
     */
    inline vec8f div(const vec8f _x, const vec8f _y) { return _mm256_div_ps(_x, _y); }

    /**
     * @brief Broadcast a scalar value to all lanes of a SIMD vector.
     *
     * @param value Scalar value to broadcast.
    */
    inline vec8f set(const f32 value) { return _mm256_set1_ps(value); }

    /**
     * @brief Create a SIMD vector initialized to zero.
     */
    inline vec8f zero() { return _mm256_setzero_ps(); }

    /**
     * @brief Permute elements within each 128-bit lane of a SIMD vector.
     *
     * @tparam dest Immediate control mask.
     * @param _x Input SIMD vector.
    */
    template<i32 dest>
    vec8f permute(const vec8f _x) { return _mm256_permute_ps(_x, dest); }

    /**
     * @brief Shuffle elements from two SIMD vectors within 128-bit lanes.
     *
     * @tparam dest Immediate shuffle control mask.
     * @param _x First input vector.
     * @param _y Second input vector.
     */
    template<i32 dest>
    vec8f shuffle(const vec8f _x, const vec8f _y) { return _mm256_shuffle_ps(_x, _y, dest); }

    /**
     * @brief Compute element-wise square root.
     */
    inline vec8f sqrt(const vec8f _x) { return _mm256_sqrt_ps(_x); }

    /**
     * @brief Compute element-wise reciprocal square root (approximate).
     */
    inline vec8f rsqrt(const vec8f _x) { return _mm256_rsqrt_ps(_x); }

    /**
     * @brief Fused multiply-add: (_x * _y) + _z.
     */
    inline vec8f fma(const vec8f _x, const vec8f _y, const vec8f _z) { return _mm256_fmadd_ps(_x, _y, _z); }

    /**
     * @brief Compute element-wise natural logarithm.
     *
     * @note Requires SVML support.
     */
    inline vec8f log(const vec8f _x) { return _mm256_log_ps(_x); }

    /**
     * @brief Compute element-wise exponential.
     *
     * @note Requires SVML support.
     */
    inline vec8f exp(const vec8f _x) { return _mm256_exp_ps(_x); }

    /**
     * @brief Compute element-wise power function.
     *
     * @note Requires SVML support.
     */
    inline vec8f pow(const vec8f _x, const vec8f _y) { return _mm256_pow_ps(_x, _y); }

    /**
     * @brief Compare _x > _y (ordered, non-signaling).
     */
    inline vec8f cmp_gt(const vec8f _x, const vec8f _y) { return _mm256_cmp_ps(_x, _y, _CMP_GT_OS); }

    /**
     * @brief Compare _x < _y (ordered, non-signaling).
     */
    inline vec8f cmp_lt(const vec8f _x, const vec8f _y) { return _mm256_cmp_ps(_x, _y, _CMP_LT_OS); }

    /**
     * @brief Compare _x == _y (ordered, non-signaling).
     */
    inline vec8f cmp_eq(const vec8f _x, const vec8f _y) { return _mm256_cmp_ps(_x, _y, _CMP_EQ_OS); }

    /**
     * @brief Blend elements from two vectors using a mask.
     *
     * @param _x Vector used when mask bit is zero.
     * @param _y Vector used when mask bit is set.
     * @param mask Selection mask.
     */
    inline vec8f blendv(const vec8f _x, const vec8f _y, const vec8f mask) { return _mm256_blendv_ps(_x, _y, mask); }

    inline vec8f where(const vec8f _x, const vec8f _y, const vec8f _z) { return blendv(_z, _y, _x); }

    inline vec8f clamp(const vec8f _x, const f32 lo, const f32 hi) { return _mm256_min_ps(_mm256_max_ps(_x, set(lo)), set(hi)); }

    inline void prefetch(const f32* _x) { _mm_prefetch(reinterpret_cast<const char*>(_x), _MM_HINT_T0); }

    /**
     * @brief Compute absolute value of each element.
     */
    inline vec8f abs(const vec8f _x) {
        const vec8f mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        return _mm256_and_ps(_x, mask);
    }

    /**
     * @brief Compute horizontal sum of all elements.
     *
     * @param _x Input SIMD vector.
     * @return Sum of all 8 elements.
     */
    inline f32 hsum(const vec8f _x) {
        const vec4f hi = _mm256_extractf128_ps(_x, 1);
        const vec4f lo = _mm256_castps256_ps128(_x);
        vec4f sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }

    /**
     * @brief Compute horizontal maximum of all elements.
     */
    inline f32 hmax(const vec8f _x) {
        const vec4f hi = _mm256_extractf128_ps(_x, 1);
        const vec4f lo = _mm256_castps256_ps128(_x);
        vec4f max = _mm_max_ps(lo, hi);
        max = _mm_max_ps(max, _mm_movehl_ps(max, max));
        max = _mm_max_ss(max, _mm_shuffle_ps(max, max, 1));
        return _mm_cvtss_f32(max);
    }

    /**
     * @brief Compute horizontal minimum of all elements.
     */
    inline f32 hmin(const vec8f _x) {
        const vec4f hi = _mm256_extractf128_ps(_x, 1);
        const vec4f lo = _mm256_castps256_ps128(_x);
        vec4f min_val = _mm_min_ps(lo, hi);
        min_val = _mm_min_ps(min_val, _mm_movehl_ps(min_val, min_val));
        min_val = _mm_min_ss(min_val, _mm_shuffle_ps(min_val, min_val, 1));
        return _mm_cvtss_f32(min_val);
    }

    inline f32 mean(const vec8f x) { return hsum(x) * (1.0f / 8.0f); }

} // namespace cortex::_fw::avx2

#endif // CORTEXMIND_CORE_ENGINE_AVX_FUNCS_HPP
