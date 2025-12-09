//
// Created by muham on 5.12.2025.
//

#ifndef CORTEXMIND_AVX2_HPP
#define CORTEXMIND_AVX2_HPP

#include <immintrin.h>
#include <algorithm>

namespace cortex::_fw::avx {
    #if defined(__AVX512F__)
        using mask8 = __mmask8;
    #elif defined(__AVX2__)
        using mask8 = uint8_t;
    #else
        #error "AVX2 or AVX-512 support is required"
    #endif

    using reg = __m256;
    using regd = __m512d;
    using float32 = float;

    inline reg load(const float32* ptr) noexcept { return _mm256_load_ps(ptr); }
    inline void store(float32* ptr, const reg v) noexcept { _mm256_store_ps(ptr, v); }

    inline reg load_u(const float32* ptr) noexcept { return _mm256_loadu_ps(ptr); }
    inline void store_u(float32* ptr, const reg v) noexcept { _mm256_storeu_ps(ptr, v); }

    inline reg load_partial(const float32* ptr, const size_t n) noexcept {
        alignas(32) float32 tmp[8] = {};
        for (size_t i = 0; i < n; ++i) tmp[i] = ptr[i];
        return load(tmp);
    }

    inline void store_partial(float32* ptr, const reg v, const size_t n) noexcept {
        alignas(32) float32 tmp[8];
        store(tmp, v);
        for (size_t i = 0; i < n; ++i) ptr[i] = tmp[i];
    }

    inline reg add(const reg a, const reg b) noexcept   { return _mm256_add_ps(a, b); }
    inline reg sub(const reg a, const reg b) noexcept   { return _mm256_sub_ps(a, b); }
    inline reg mul(const reg a, const reg b) noexcept   { return _mm256_mul_ps(a, b); }
    inline reg div(const reg a, const reg b) noexcept   { return _mm256_div_ps(a, b); }
    inline reg fma(const reg a, const reg b, const reg c) noexcept { return _mm256_fmadd_ps(a, b, c); }

    inline reg neg(const reg a) noexcept { return _mm256_xor_ps(a, _mm256_set1_ps(-0.0f)); }
    inline reg abs(const reg a) noexcept { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a); }

    inline reg broadcast(const float32 x) noexcept    { return _mm256_set1_ps(x); }
    inline reg zero() noexcept { return _mm256_setzero_ps(); }

    inline reg sqrt(const reg x) noexcept { return _mm256_sqrt_ps(x); }
    inline reg r_sqrt(const reg x) noexcept { return _mm256_rsqrt_ps(x); }
    inline reg exp_approx(reg x) noexcept {
        const reg max_x = _mm256_set1_ps(88.722839f);
        const reg min_x = _mm256_set1_ps(-103.972084f);

        x = _mm256_min_ps(max_x, _mm256_max_ps(min_x, x));

        const reg ln2     = _mm256_set1_ps(0.6931471805599453f);
        const reg inv_ln2 = _mm256_set1_ps(1.4426950408889634f);

        const reg y = _mm256_mul_ps(x, inv_ln2);

        const __m256i yi = _mm256_cvttps_epi32(y);
        const reg y_floor = _mm256_cvtepi32_ps(yi);

        const reg r = _mm256_sub_ps(x, _mm256_mul_ps(y_floor, ln2));

        const reg c1 = _mm256_set1_ps(1.0f);
        const reg c2 = _mm256_set1_ps(0.5f);
        const reg c3 = _mm256_set1_ps(1.0f / 6.0f);
        const reg c4 = _mm256_set1_ps(1.0f / 24.0f);
        const reg c5 = _mm256_set1_ps(1.0f / 120.0f);

        const reg r2 = _mm256_mul_ps(r, r);
        const reg r4 = _mm256_mul_ps(r2, r2);

        reg p1 = _mm256_fmadd_ps(c3, r, c2);
        p1 = _mm256_fmadd_ps(p1, r, c1);

        const reg p2 = _mm256_fmadd_ps(c5, r, c4);

        reg poly = _mm256_fmadd_ps(p2, r4, p1);
        poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(1.0f));

        const __m256i bias = _mm256_set1_epi32(127);
        __m256i ei = _mm256_add_epi32(yi, bias);
        ei = _mm256_slli_epi32(ei, 23);
        const reg pow2n = _mm256_castsi256_ps(ei);

        return _mm256_mul_ps(poly, pow2n);
    }

    inline reg log_approx(const reg x) noexcept {
        const reg zero = _mm256_set1_ps(0.0f);
        const reg one  = _mm256_set1_ps(1.0f);
        const reg ln2  = _mm256_set1_ps(0.6931471805599453f);

        //reg invalid_mask = _mm256_cmp_ps(x, zero, _CMP_LE_OQ);

        const __m256i ix = _mm256_castps_si256(x);

        const __m256i bias = _mm256_set1_epi32(127);
        __m256i exp_i = _mm256_srli_epi32(ix, 23);
        exp_i = _mm256_sub_epi32(exp_i, bias);
        const reg e = _mm256_cvtepi32_ps(exp_i);

        const __m256i mask = _mm256_set1_epi32(0x007FFFFF);
        const __m256i one_bits = _mm256_set1_epi32(0x3f800000);
        const __m256i mant_i = _mm256_or_si256(_mm256_and_si256(ix, mask), one_bits);

        const reg m = _mm256_castsi256_ps(mant_i);
        const reg r = _mm256_sub_ps(m, one);

        const reg r2 = _mm256_mul_ps(r, r);
        const reg r4 = _mm256_mul_ps(r2, r2);

        const reg c1 = _mm256_set1_ps(-0.5f);
        const reg c2 = _mm256_set1_ps(1.0f / 3.0f);
        const reg c3 = _mm256_set1_ps(-0.25f);
        const reg c4 = _mm256_set1_ps(0.2f);

        const reg p1 = _mm256_fmadd_ps(c2, r, c1);
        const reg p2 = _mm256_fmadd_ps(c4, r, c3);

        reg poly = _mm256_fmadd_ps(p2, r4, p1);
        poly = _mm256_mul_ps(poly, r);

        reg result = _mm256_fmadd_ps(e, ln2, poly);

        const reg nanv = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());
        const reg neginf = _mm256_set1_ps(-INFINITY);

        result = _mm256_blendv_ps(result, nanv, _mm256_cmp_ps(x, zero, _CMP_LT_OQ));

        result = _mm256_blendv_ps(result, neginf, _mm256_cmp_ps(x, zero, _CMP_EQ_OQ));

        return result;
    }

    inline float h_sum(const reg v) noexcept {
        const __m128 lo = _mm256_castps256_ps128(v);
        const __m128 hi = _mm256_extractf128_ps(v, 1);

        __m128 s = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(s);
        s = _mm_add_ps(s, shuf);
        shuf = _mm_movehl_ps(shuf, s);
        s = _mm_add_ss(s, shuf);

        return _mm_cvtss_f32(s);
    }

    inline float32 max(const reg v) noexcept {
        const __m128 hi = _mm256_extractf128_ps(v, 1);
        const __m128 lo = _mm256_castps256_ps128(v);
        __m128 m = _mm_max_ps(hi, lo);
        m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(0,0,3,2)));
        m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(0,0,0,1)));
        return _mm_cvtss_f32(m);
    }

    inline float32 min(const reg v) noexcept {
        const __m128 hi = _mm256_extractf128_ps(v, 1);
        const __m128 lo = _mm256_castps256_ps128(v);
        __m128 m = _mm_min_ps(hi, lo);
        m = _mm_min_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(0,0,3,2)));
        m = _mm_min_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(0,0,0,1)));
        return _mm_cvtss_f32(m);
    }

    inline void add_kernel(const float32* a, const float32* b, float32* out, const size_t n) noexcept {
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            const reg va = load(a + i);
            const reg vb = load(b + i);
            store(out + i, add(va, vb));
        }
        if (i < n) {
            const reg va = load_partial(a + i, n - i);
            const reg vb = load_partial(b + i, n - i);
            store_partial(out + i, add(va, vb), n - i);
        }
    }

    inline void mul_kernel(const float32* a, const float32* b, float32* out, const size_t n) noexcept {
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            const reg va = load(a + i);
            const reg vb = load(b + i);
            store(out + i, mul(va, vb));
        }
        if (i < n) {
            const reg va = load_partial(a + i, n - i);
            const reg vb = load_partial(b + i, n - i);
            store_partial(out + i, mul(va, vb), n - i);
        }
    }

    inline void sub_kernel(const float32* a, const float32* b, float32* out, const size_t n) noexcept {
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            const reg va = load(a + i);
            const reg vb = load(b + i);
            store(out + i, sub(va, vb));
        }
        if (i < n) {
            const reg va = load_partial(a + i, n - i);
            const reg vb = load_partial(b + i, n - i);
            store_partial(out + i, sub(va, vb), n - i);
        }
    }

    inline void div_kernel(const float32* a, const float32* b, float32* out, const size_t n) noexcept {
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            const reg va = load(a + i);
            const reg vb = load(b + i);
            store(out + i, div(va, vb));
        }
        if (i < n) {
            const reg va = load_partial(a + i, n - i);
            const reg vb = load_partial(b + i, n - i);
            store_partial(out + i, div(va, vb), n - i);
        }
    }

    inline void fma_kernel(const float32* a, const float32* b, float32* out, const size_t n) noexcept {
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            const reg va = load(a + i);
            const reg vb = load(b + i);
            const reg vc = load(out + i);
            store(out + i, fma(va, vb, vc));
        }
        if (i < n) {
            const reg va = load_partial(a + i, n - i);
            const reg vb = load_partial(b + i, n - i);
            const reg vc = load_partial(out + i, n - i);
            store_partial(out + i, fma(va, vb, vc), n - i);
        }
    }

    inline void matmul_kernel(const float* A, const float* B, float* C, const size_t M, const size_t N, const size_t K) noexcept {
        for (size_t i = 0; i < M; i += 8) {
            const size_t im = std::min(M - i, static_cast<size_t>(8));

            for (size_t j = 0; j < N; j += 8) {
                const size_t jm = std::min(N - j, static_cast<size_t>(8));

                reg acc[8][8];
                for (size_t ii = 0; ii < im; ++ii) {
                    for (size_t jj = 0; jj < jm; ++jj) {
                        acc[ii][jj] = _mm256_setzero_ps();
                    }
                }

                for (size_t k = 0; k < K; ++k) {
                    const reg b = _mm256_loadu_ps(&B[k*N + j]);

                    for (size_t ii = 0; ii < im; ++ii) {
                        const reg a = _mm256_set1_ps(A[(i+ii)*K + k]);

                        for (size_t jj = 0; jj < jm; ++jj) {
                            acc[ii][jj] = _mm256_fmadd_ps(a, _mm256_permute_ps(b, _MM_SHUFFLE(jj,jj,jj,jj)), acc[ii][jj]);
                        }
                    }
                }

                for (size_t ii = 0; ii < im; ++ii) {
                    for (size_t jj = 0; jj < jm; ++jj) {
                        _mm256_storeu_ps(&C[(i+ii)*N + j + jj], acc[ii][jj]);
                    }
                }
            }
        }
    }

}

#endif //CORTEXMIND_AVX2_HPP