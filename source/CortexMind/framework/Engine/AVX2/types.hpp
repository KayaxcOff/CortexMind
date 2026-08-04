//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_TYPES_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_TYPES_HPP

#include <tlx/concepts.hpp>
#include <tlx/types.hpp>
#include <immintrin.h>

namespace cortex::_fw::avx2 {
    /**
     * @brief Base interface for SIMD vector types.
     *
     * Provides a common interface shared by all SIMD vector wrappers
     * used by the CortexMind execution engine.
     *
     * @tparam T Scalar element type.
     */
    template<tlx::arithmetic_like T>
    struct VecBase {
        virtual ~VecBase() = default;
        using value_type = T;
    };

    /**
     * @brief AVX2 vector containing eight single-precision floating-point values.
     *
     * This class wraps a native AVX2 register and provides arithmetic,
     * comparison, and storage operations through a C++ interface.
     */
    struct vec8f : VecBase<float> {
        vec8f() {
            this->reg = _mm256_setzero_ps();
        }
        explicit vec8f(const float* src) {
            this->reg = _mm256_load_ps(src);
        }
        explicit vec8f(const __m256 src) {
            this->reg = src;
        }
        vec8f(const vec8f&) = default;
        vec8f(vec8f&&) noexcept = default;
        ~vec8f() override = default;

        [[nodiscard]]
        __m256& raw() {
            return this->reg;
        }
        [[nodiscard]]
        const __m256& raw() const {
            return this->reg;
        }

        void store(value_type* dst) const {
            _mm256_store_ps(dst, this->reg);
        }

        vec8f operator+(const vec8f& other) const {
            return vec8f(_mm256_add_ps(this->reg, other.reg));
        }
        vec8f operator-(const vec8f& other) const {
            return vec8f(_mm256_sub_ps(this->reg, other.reg));
        }
        vec8f operator*(const vec8f& other) const {
            return vec8f(_mm256_mul_ps(this->reg, other.reg));
        }
        vec8f operator/(const vec8f& other) const {
            return vec8f(_mm256_div_ps(this->reg, other.reg));
        }

        vec8f& operator+=(const vec8f& other) {
            *this = *this + other;
            return *this;
        }
        vec8f& operator-=(const vec8f& other) {
            *this = *this - other;
            return *this;
        }
        vec8f& operator*=(const vec8f& other) {
            *this = *this * other;
            return *this;
        }
        vec8f& operator/=(const vec8f& other) {
            *this = *this / other;
            return *this;
        }

        vec8f operator<(const vec8f& other) const {
            return vec8f(_mm256_cmp_ps(this->reg, other.reg, _CMP_LT_OQ));
        }
        vec8f operator>(const vec8f& other) const {
            return vec8f(_mm256_cmp_ps(this->reg, other.reg, _CMP_GT_OQ));
        }
        vec8f operator<=(const vec8f& other) const {
            return vec8f(_mm256_cmp_ps(this->reg, other.reg, _CMP_LE_OQ));
        }
        vec8f operator>=(const vec8f& other) const {
            return vec8f(_mm256_cmp_ps(this->reg, other.reg, _CMP_GE_OQ));
        }
        vec8f operator!=(const vec8f& other) const {
            return vec8f(_mm256_cmp_ps(this->reg, other.reg, _CMP_NEQ_OQ));
        }
        vec8f operator==(const vec8f& other) const {
            return vec8f(_mm256_cmp_ps(this->reg, other.reg, _CMP_EQ_OQ));
        }

        vec8f& operator=(const vec8f&) = default;
        vec8f& operator=(vec8f&&) noexcept = default;
    private:
        __m256 reg{};
    };

    /**
     * @brief AVX2 vector containing eight 32-bit signed integers.
     */
    struct vec8i : VecBase<std::int32_t> {
        vec8i() {
            this->reg = _mm256_setzero_si256();
        }
        explicit vec8i(const std::int32_t* src) {
            this->reg = _mm256_load_si256(reinterpret_cast<const __m256i*>(src));
        }
        explicit vec8i(const __m256i src) {
            this->reg = src;
        }
        vec8i(const vec8i&) = default;
        vec8i(vec8i&&) noexcept = default;
        ~vec8i() override = default;

        [[nodiscard]]
        const __m256i& raw() {
            return this->reg;
        }
        [[nodiscard]]
        const __m256i& raw() const {
            return this->reg;
        }

        void store(value_type* dst) const {
            _mm256_store_si256(reinterpret_cast<__m256i *>(dst), this->reg);
        }

        vec8i operator+(const vec8i& other) const {
            return vec8i(_mm256_add_epi32(this->reg, other.reg));
        }
        vec8i operator-(const vec8i& other) const {
            return vec8i(_mm256_sub_epi32(this->reg, other.reg));
        }
        vec8i operator*(const vec8i& other) const {
            return vec8i(_mm256_mullo_epi32(this->reg, other.reg));
        }
        vec8i operator/(const vec8i& other) const {
            return vec8i(_mm256_div_epi32(this->reg, other.reg));
        }

        vec8i& operator+=(const vec8i& other) {
            *this = *this + other;
            return *this;
        }
        vec8i& operator-=(const vec8i& other) {
            *this = *this - other;
            return *this;
        }
        vec8i& operator*=(const vec8i& other) {
            *this = *this * other;
            return *this;
        }
        vec8i& operator/=(const vec8i& other) {
            *this = *this / other;
            return *this;
        }

        vec8i operator<(const vec8i& other) const {
            return vec8i(_mm256_cmpgt_epi32(this->reg, other.reg));
        }
        vec8i operator>(const vec8i& other) const {
            return vec8i(_mm256_cmpgt_epi32(other.reg, this->reg));
        }
        vec8i operator<=(const vec8i& other) const {
            const auto lt = _mm256_cmpgt_epi32(this->reg, other.reg);

            return vec8i(_mm256_xor_si256(lt, _mm256_set1_epi32(-1)));
        }
        vec8i operator>=(const vec8i& other) const {
            const auto lt = _mm256_cmpgt_epi32(other.reg, this->reg);

            return vec8i(_mm256_xor_si256(lt, _mm256_set1_epi32(-1)));
        }
        vec8i operator!=(const vec8i& other) const {
            const auto eq = _mm256_cmpeq_epi32(this->reg, other.reg);

            return vec8i(_mm256_xor_si256(eq,_mm256_set1_epi32(-1)));
        }
        vec8i operator==(const vec8i& other) const {
            return vec8i(_mm256_cmpeq_epi32(this->reg, other.reg));
        }

        vec8i& operator=(const vec8i&) = default;
        vec8i& operator=(vec8i&&) noexcept = default;
    private:
        __m256i reg{};
    };

    /**
     * @brief AVX2 vector containing four double-precision floating-point values.
     *
     * Wraps a native AVX2 register (`__m256d`) and provides arithmetic,
     * comparison, and memory operations through a lightweight C++ interface.
     *
     * The class is intended for use by the CortexMind AVX2 execution engine
     * and serves as the SIMD representation of four `double` values.
     */
    struct vec4d : VecBase<double> {
        vec4d() {
            this->reg = _mm256_setzero_pd();
        }
        explicit vec4d(const double* src) {
            this->reg = _mm256_load_pd(src);
        }
        explicit vec4d(const __m256d src) {
            this->reg = src;
        }
        vec4d(const vec4d&) = default;
        vec4d(vec4d&&) noexcept = default;
        ~vec4d() override = default;

        [[nodiscard]]
        __m256d& raw() {
            return this->reg;
        }
        [[nodiscard]]
        const __m256d& raw() const {
            return this->reg;
        }

        void store(value_type* dst) const {
            _mm256_store_pd(dst, this->reg);
        }

        vec4d operator+(const vec4d& other) const {
            return vec4d(_mm256_add_pd(this->reg, other.reg));
        }
        vec4d operator-(const vec4d& other) const {
            return vec4d(_mm256_sub_pd(this->reg, other.reg));
        }
        vec4d operator*(const vec4d& other) const {
            return vec4d(_mm256_mul_pd(this->reg, other.reg));
        }
        vec4d operator/(const vec4d& other) const {
            return vec4d(_mm256_div_pd(this->reg, other.reg));
        }

        vec4d& operator+=(const vec4d& other) {
            *this = *this + other;
            return *this;
        }
        vec4d& operator-=(const vec4d& other) {
            *this = *this - other;
            return *this;
        }
        vec4d& operator*=(const vec4d& other) {
            *this = *this * other;
            return *this;
        }
        vec4d& operator/=(const vec4d& other) {
            *this = *this / other;
            return *this;
        }

        vec4d operator<(const vec4d& other) const {
            return vec4d(_mm256_cmp_pd(this->reg, other.reg, _CMP_LT_OQ));
        }
        vec4d operator>(const vec4d& other) const {
            return vec4d(_mm256_cmp_pd(this->reg, other.reg, _CMP_GT_OQ));
        }
        vec4d operator<=(const vec4d& other) const {
            return vec4d(_mm256_cmp_pd(this->reg, other.reg, _CMP_LE_OQ));
        }
        vec4d operator>=(const vec4d& other) const {
            return vec4d(_mm256_cmp_pd(this->reg, other.reg, _CMP_GE_OQ));
        }
        vec4d operator!=(const vec4d& other) const {
            return vec4d(_mm256_cmp_pd(this->reg, other.reg, _CMP_EQ_OQ));
        }
        vec4d operator==(const vec4d& other) const {
            return vec4d(_mm256_cmp_pd(this->reg, other.reg, _CMP_EQ_OQ));
        }

        vec4d& operator=(const vec4d&) = default;
        vec4d& operator=(vec4d&&) noexcept = default;
    private:
        __m256d reg{};
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_TYPES_HPP