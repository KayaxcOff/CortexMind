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
        }

        [[nodiscard]]
        __forceinline vec8f load(const float* src) const {
            return vec8f(_mm256_maskload_ps(src, mask8(this->m_size).raw()));
        }
        [[nodiscard]]
        __forceinline vec4d load(const double* src) const {
            return vec4d(_mm256_maskload_pd(src, mask4(this->m_size).raw()));
        }
        [[nodiscard]]
        __forceinline vec8i load(const std::int32_t* src) const {
            return vec8i(_mm256_maskload_epi32(src, mask8(this->m_size).raw()));
        }

        __forceinline void store(float* dst, const vec8f& src) const {
            _mm256_maskstore_ps(dst, mask8(this->m_size).raw(), src.raw());
        }
        __forceinline void store(double* dst, const vec4d& src) const {
            _mm256_maskstore_pd(dst, mask4(this->m_size).raw(), src.raw());
        }
        __forceinline void store(std::int32_t* dst, const vec8i& src) const {
            _mm256_maskstore_epi32(dst, mask8(this->m_size).raw(), src.raw());
        }

        [[nodiscard]]
        std::size_t size() const noexcept;
    private:
        std::size_t m_size;
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_MEMORY_MASK_RUNTIME_HPP