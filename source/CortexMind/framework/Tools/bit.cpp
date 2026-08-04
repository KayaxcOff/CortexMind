//
// Created by muham on 4.08.2026.
//

#include "CortexMind/framework/Tools/bit.hpp"
#include <CortexMind/framework/Engine/AVX2/types.hpp>

using namespace cortex::_fw;

void detail::load(const tlx::bfloat16 *src, native<avx2::vec8f> &v) {
    alignas(32) float lo[8];
    alignas(32) float hi[8];

    for (std::size_t i = 0; i < 8; ++i) {
        lo[i] = static_cast<float>(src[i]);
    }

    for (std::size_t i = 0; i < 8; ++i) {
        hi[i] = static_cast<float>(src[i + 8]);
    }

    v.low = avx2::vec8f(lo);
    v.high = avx2::vec8f(hi);
}

void detail::store(tlx::bfloat16 *dst, const native<avx2::vec8f> &v) {
    alignas(32) float lo[8];
    alignas(32) float hi[8];

    v.low.store(lo);
    v.high.store(hi);

    for (std::size_t i = 0; i < 8; ++i) {
        dst[i] = tlx::bfloat16(lo[i]);
    }

    for (std::size_t i = 0; i < 8; ++i) {
        dst[i + 8] = tlx::bfloat16(hi[i]);
    }
}

void detail::load(const tlx::half *src, native<avx2::vec8f> &v) {
    const auto* bits = reinterpret_cast<const std::uint16_t*>(src);

    const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bits));

    const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bits + 8));

    v.low  = avx2::vec8f(_mm256_cvtph_ps(lo));
    v.high = avx2::vec8f(_mm256_cvtph_ps(hi));
}

void detail::store(tlx::half *dst, const native<avx2::vec8f> &v) {
    auto* bits = reinterpret_cast<std::uint16_t*>(dst);

    const __m128i lo = _mm256_cvtps_ph(v.low.raw(), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    const __m128i hi = _mm256_cvtps_ph(v.high.raw(), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(bits), lo);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(bits + 8), hi);
}