//
// Created by muham on 4.08.2026.
//

#include "CortexMind/framework/Tools/cast.hpp"
#include <immintrin.h>

void cortex::_fw::convert(float *dst, const tlx::bfloat16 *src, const std::size_t size) {
    const auto* raw = reinterpret_cast<const std::uint16_t*>(src);

    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        const __m128i bits16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(raw + i));

        const __m256i widened = _mm256_cvtepu16_epi32(bits16);
        const __m256i shifted = _mm256_slli_epi32(widened, 16);
        const __m256  f32     = _mm256_castsi256_ps(shifted);

        _mm256_storeu_ps(dst + i, f32);
    }

    for (; i < size; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

void cortex::_fw::convert(float *dst, const tlx::half *src, const std::size_t size) {
    const auto* raw = reinterpret_cast<const std::uint16_t*>(src);

    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        const __m128i bits16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(raw + i));

        const __m256 f32 = _mm256_cvtph_ps(bits16);
        _mm256_storeu_ps(dst + i, f32);
    }

    for (; i < size; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

void cortex::_fw::convert(tlx::bfloat16 *dst, const float *src, const std::size_t size) {
    auto* raw = reinterpret_cast<std::uint16_t*>(dst);

    const __m256i bias      = _mm256_set1_epi32(0x7FFF);
    const __m256i one       = _mm256_set1_epi32(1);

    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        const __m256  f32  = _mm256_loadu_ps(src + i);
        const __m256i bits = _mm256_castps_si256(f32);

        const __m256i lsb  = _mm256_and_si256(_mm256_srli_epi32(bits, 16), one);

        const __m256i rounded = _mm256_add_epi32(bits, _mm256_add_epi32(bias, lsb));

        const __m256i shifted = _mm256_srli_epi32(rounded, 16);

        const __m128i lo = _mm256_castsi256_si128(shifted);
        const __m128i hi = _mm256_extracti128_si256(shifted, 1);
        const __m128i packed = _mm_packus_epi32(lo, hi);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(raw + i), packed);
    }

    for (; i < size; ++i) {
        dst[i] = src[i];
    }
}

void cortex::_fw::convert(tlx::half *dst, const float *src, const std::size_t size) {
    auto* raw = reinterpret_cast<std::uint16_t*>(dst);

    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        const __m256 f32 = _mm256_loadu_ps(src + i);
        const __m128i h  = _mm256_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(raw + i), h);
    }

    for (; i < size; ++i) {
        dst[i] = src[i];
    }
}
