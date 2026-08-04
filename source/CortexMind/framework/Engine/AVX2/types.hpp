//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_TYPES_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_TYPES_HPP

#include <CortexMind/framework/Tools/bit.hpp>
#include <CortexMind/framework/Tools/native.hpp>
#include <tlx/concepts.hpp>
#include <tlx/types.hpp>
#include <immintrin.h>

namespace cortex::_fw::avx2 {
    template<tlx::arithmetic_like T>
    struct VecBase {
        virtual ~VecBase() = default;

        using type = T;

        virtual void store(type* dst) const = 0;
    };

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
        __m256 raw() {
            return this->reg;
        }
        [[nodiscard]]
        __m256 raw() const {
            return this->reg;
        }

        void store(type* dst) const override {
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
        __m256i raw() {
            return this->reg;
        }
        [[nodiscard]]
        __m256i raw() const {
            return this->reg;
        }

        void store(type *dst) const override {
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

    struct vec16bf : VecBase<tlx::bfloat16> {
        vec16bf() = default;
        explicit vec16bf(const tlx::bfloat16* src) {
            detail::load_bf16(src, this->rlow, this->rhigh);
        }
        vec16bf(const vec8f &low, const vec8f &high) {
            this->rlow = low;
            this->rhigh = high;
        }
        vec16bf(const vec16bf&) = default;
        vec16bf(vec16bf&&) noexcept = default;
        ~vec16bf() override = default;

        [[nodiscard]]
        vec8f& low() {
            return this->rlow;
        }
        [[nodiscard]]
        const vec8f& low() const {
            return this->rlow;
        }
        [[nodiscard]]
        vec8f& high() {
            return this->rhigh;
        }
        [[nodiscard]]
        const vec8f& high() const {
            return this->rhigh;
        }

        [[nodiscard]]
        native<vec8f>& raw() {
            return this->reg;
        }
        [[nodiscard]]
        const native<vec8f>& raw() const {
            return this->reg;
        }

        void store(type *dst) const override {
            detail::store_bf16(dst, this->rlow, this->rhigh);
        }

        vec16bf operator+(const vec16bf& other) const {
            return {
                this->rlow + other.rlow,
                this->rhigh + other.rhigh
            };
        }
        vec16bf operator-(const vec16bf& other) const {
            return {
                this->rlow - other.rlow,
                this->rhigh - other.rhigh
            };
        }
        vec16bf operator*(const vec16bf& other) const {
            return {
                this->rlow * other.rlow,
                this->rhigh * other.rhigh
            };
        }
        vec16bf operator/(const vec16bf& other) const {
            return {
                this->rlow / other.rlow,
                this->rhigh / other.rhigh
            };
        }

        vec16bf& operator+=(const vec16bf& other) {
            *this = *this + other;
            return *this;
        }
        vec16bf& operator-=(const vec16bf& other) {
            *this = *this - other;
            return *this;
        }
        vec16bf& operator*=(const vec16bf& other) {
            *this = *this * other;
            return *this;
        }
        vec16bf& operator/=(const vec16bf& other) {
            *this = *this / other;
            return *this;
        }

        vec16bf operator<(const vec16bf& other) const {
            return {
                this->rlow < other.rlow,
                this->rhigh < other.rhigh
            };
        }
        vec16bf operator>(const vec16bf& other) const {
            return {
                this->rlow > other.rlow,
                this->rhigh > other.rhigh
            };
        }
        vec16bf operator<=(const vec16bf& other) const {
            return {
                this->rlow <= other.rlow,
                this->rhigh <= other.rhigh
            };
        }
        vec16bf operator>=(const vec16bf& other) const {
            return {
                this->rlow >= other.rlow,
                this->rhigh >= other.rhigh
            };
        }
        vec16bf operator!=(const vec16bf& other) const {
            return {
                this->rlow != other.rlow,
                this->rhigh != other.rhigh
            };
        }
        vec16bf operator==(const vec16bf& other) const {
            return {
                this->rlow == other.rlow,
                this->rhigh == other.rhigh
            };
        }

        vec16bf& operator=(const vec16bf&) = default;
        vec16bf& operator=(vec16bf&&) noexcept = default;
    private:
        vec8f rlow{};
        vec8f rhigh{};
        native<vec8f> reg;
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_TYPES_HPP