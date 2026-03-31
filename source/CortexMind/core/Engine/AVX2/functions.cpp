//
// Created by muham on 29.03.2026.
//

#include "CortexMind/core/Engine/AVX2/functions.hpp"
#include <CortexMind/core/Engine/AVX2/cmp.hpp>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

vec8f avx2::load(const f32 *src) {
    return _mm256_load_ps(src);
}

void avx2::store(f32 *dst, const vec8f src) {
    _mm256_store_ps(dst, src);
}

vec8f avx2::loadu(const f32 *src) {
    return _mm256_loadu_ps(src);
}

void avx2::storeu(f32 *dst, const vec8f src) {
    _mm256_storeu_ps(dst, src);
}

vec8f avx2::set1(const f32 value) {
    return _mm256_set1_ps(value);
}

vec8f avx2::zero() {
    return _mm256_setzero_ps();
}

vec8f avx2::add(const vec8f Xx, const vec8f Xy) {
    return _mm256_add_ps(Xx, Xy);
}

vec8f avx2::sub(const vec8f Xx, const vec8f Xy) {
    return _mm256_sub_ps(Xx, Xy);
}

vec8f avx2::mul(const vec8f Xx, const vec8f Xy) {
    return _mm256_mul_ps(Xx, Xy);
}

vec8f avx2::div(const vec8f Xx, const vec8f Xy) {
    return _mm256_div_ps(Xx, Xy);
}

vec8f avx2::exp(const vec8f Xx) {
    return _mm256_exp_ps(Xx);
}

vec8f avx2::log(const vec8f Xx) {
    return _mm256_log_ps(Xx);
}

vec8f avx2::sqrt(const vec8f Xx) {
    return _mm256_sqrt_ps(Xx);
}

vec8f avx2::abs(const vec8f Xx) {
    return _mm256_andnot_ps(set1(-0.0f), Xx);
}

vec8f avx2::neg(const vec8f Xx) {
    const vec8f sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<i32>(0x80000000u)));
    return _mm256_xor_ps(Xx, sign_mask);
}

vec8f avx2::tanh(const vec8f Xx) {
    return _mm256_tanh_ps(Xx);
}

vec8f avx2::pow(const vec8f Xx, const vec8f Xy) {
    return _mm256_pow_ps(Xx, Xy);
}

vec8f avx2::fma(const vec8f Xx, const vec8f Xy, const vec8f Xz) {
    return _mm256_fmadd_ps(Xx, Xy, Xz);
}

vec8f avx2::rcp_nr(const vec8f Xx) {
    const vec8f r0 = _mm256_rcp_ps(Xx);
    return _mm256_mul_ps(r0, _mm256_fnmadd_ps(Xx, r0, set1(2.0f)));
}

vec8f avx2::blendv(const vec8f Xx, const vec8f Xy, const vec8f Xz) {
    return _mm256_blendv_ps(Xx, Xy, Xz);
}

vec8f avx2::min(const vec8f Xx, const vec8f Xy) {
    return _mm256_min_ps(Xx, Xy);
}

vec8f avx2::max(const vec8f Xx, const vec8f Xy) {
    return _mm256_max_ps(Xx, Xy);
}

f32 avx2::hsum(const vec8f Xx) {
    const vec4f hi = _mm256_extractf128_ps(Xx, 1);
    const vec4f lo = _mm256_castps256_ps128(Xx);
    vec4f sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

f32 avx2::hmax(const vec8f Xx) {
    const vec4f hi = _mm256_extractf128_ps(Xx, 1);
    const vec4f lo = _mm256_castps256_ps128(Xx);
    vec4f max = _mm_max_ps(lo, hi);
    max = _mm_max_ps(max, _mm_movehl_ps(max, max));
    max = _mm_max_ss(max, _mm_shuffle_ps(max, max, 1));
    return _mm_cvtss_f32(max);
}

f32 avx2::hmin(const vec8f Xx) {
    const vec4f hi = _mm256_extractf128_ps(Xx, 1);
    const vec4f lo = _mm256_castps256_ps128(Xx);
    vec4f min_val = _mm_min_ps(lo, hi);
    min_val = _mm_min_ps(min_val, _mm_movehl_ps(min_val, min_val));
    min_val = _mm_min_ss(min_val, _mm_shuffle_ps(min_val, min_val, 1));
    return _mm_cvtss_f32(min_val);
}

vec8f avx2::relu(const vec8f Xx) {
    return max(zero(), Xx);
}

vec8f avx2::leaky_relu(const vec8f Xx, const f32 alpha) {
    const vec8f mask = cmp::gt(Xx, zero());
    const vec8f neg_branch = mul(set1(alpha), Xx);
    return blendv(neg_branch, Xx, mask);
}

vec8f avx2::sigmoid(const vec8f Xx) {
    const vec8f neg_x = neg(Xx);
    const vec8f exp_neg = exp(neg_x);
    const vec8f denom = add(set1(1.0f), exp_neg);
    return div(set1(1.0f), denom);
}

vec8f avx2::sigmoid_fast(const vec8f Xx) {
    const vec8f exp_neg = avx2::exp(neg(Xx));
    return rcp_nr(add(set1(1.0f), exp_neg));
}

vec8f avx2::gelu_exact(const vec8f Xx) {
    const vec8f inv_sqrt2 = set1(0.7071067811865475f);
    const vec8f erf_input = mul(Xx, inv_sqrt2);
    const vec8f erf_val = _mm256_erf_ps(erf_input);
    const vec8f one_plus_erf = add(set1(1.0f), erf_val);
    return mul(mul(set1(0.5f), Xx), one_plus_erf);
}

vec8f avx2::gelu(const vec8f Xx) {
    const vec8f c0 = set1(0.7978845608028654f);
    const vec8f c1 = set1(0.044715f);

    const vec8f x3 = mul(mul(Xx, Xx), Xx);
    const vec8f inner = mul(c0, add(Xx, mul(c1, x3)));
    const vec8f tanh_val = tanh(inner);
    return mul(mul(set1(0.5f), Xx), add(set1(1.0f), tanh_val));
}

vec8f avx2::silu(const vec8f Xx) {
    return mul(Xx, sigmoid(Xx));
}

vec8f avx2::swish(const vec8f Xx, const f32 beta) {
    return mul(Xx, sigmoid(mul(set1(beta), Xx)));
}

vec8f avx2::softmax(const vec8f Xx) {
    const vec8f x_max = set1(hmax(Xx));
    const vec8f shifted = sub(Xx, x_max);
    const vec8f exp_val = exp(shifted);
    const vec8f sum = set1(hsum(exp_val));
    return div(exp_val, sum);
}