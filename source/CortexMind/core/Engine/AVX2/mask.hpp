//
// Created by muham on 7.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_MASK_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_MASK_HPP

#include <CortexMind/core/Engine/AVX2/params.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    struct mask {
        explicit mask(const size_t N) : N(N) {
            this->m_mask = this->create();
        }

        [[nodiscard]]
        __forceinline vec8f load(const f32* src) const {
            return _mm256_maskload_ps(src, this->m_mask);
        }
        __forceinline void store(f32* dst, const vec8f src) const {
            _mm256_maskstore_ps(dst, this->m_mask, src);
        }
    private:
        size_t N;
        vec8i m_mask{};

        [[nodiscard]]
        __forceinline vec8i create() const {
            alignas(32) static constexpr int32_t mask_data[16] = {
                -1, -1, -1, -1, -1, -1, -1, -1,
                 0,  0,  0,  0,  0,  0,  0,  0
            };
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask_data[8 - this->N]));
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_MASK_HPP