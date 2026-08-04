//
// Created by muham on 4.08.2026.
//

#include "CortexMind/framework/Tools/bit.hpp"
#include <CortexMind/framework/Engine/AVX2/types.hpp>

using namespace cortex::_fw;

void detail::load_bf16(const tlx::bfloat16 *src, native<avx2::vec8f> &v) {
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

void detail::store_bf16(tlx::bfloat16 *dst, const native<avx2::vec8f> &v) {
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