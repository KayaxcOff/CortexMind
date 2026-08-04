//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_MEMORY_MASK_RUNTIME_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_MEMORY_MASK_RUNTIME_HPP

#include <CortexMind/framework/Engine/AVX2/mask.hpp>
#include <CortexMind/framework/Tools/errors.hpp>

namespace cortex::_fw::avx2 {
    struct mask {
        explicit mask(const std::size_t n) {
            this->m_size = n;
            this->m_mask = Init(this->m_size);
        }

        [[nodiscard]]
        __forceinline vec8f load(const float* src) const {
            return vec8f(_mm256_maskload_ps(src, this->m_mask.raw()));
        }
        __forceinline void store(float* dst, const vec8f& src) const {
            _mm256_maskstore_ps(dst, this->m_mask.raw(), src.raw());
        }

        [[nodiscard]]
        std::size_t size() const noexcept;

        vec8i operator()() {
            return this->m_mask;
        }
    private:
        std::size_t m_size;
        vec8i m_mask;
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_MEMORY_MASK_RUNTIME_HPP