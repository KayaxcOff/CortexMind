//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_MEMORY_MASK_RUNTIME_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_MEMORY_MASK_RUNTIME_HPP

#include <CortexMind/framework/Engine/AVX2/mask.hpp>
#include <CortexMind/framework/Tools/errors.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief Runtime AVX2 masked memory operations.
     *
     * Provides masked load and store operations whose active lane count
     * is determined at runtime. This is useful when the number of valid
     * elements is not known until execution time, such as processing the
     * tail of dynamically sized tensors.
     *
     * Unlike the compile-time @ref mask template, this implementation
     * stores the active lane count internally and generates the required
     * mask on demand.
     */
    struct mask {
        /**
         * @brief Constructs a runtime mask.
         *
         * @param n Number of active lanes.
         */
        explicit mask(const std::size_t n) {
            CXM_ASSERT(n > 8, "Runtime AVX2 mask size must be in range [0, 8]");
            this->m_size = n;
        }

        /**
         * @brief Performs a masked load of single-precision floating-point values.
         *
         * Loads up to the configured number of active lanes from memory.
         *
         * @param src Source memory.
         *
         * @return Loaded SIMD vector.
         */
        [[nodiscard]]
        __forceinline vec8f load(const float* src) const {
            return vec8f(_mm256_maskload_ps(src, mask8(this->m_size)));
        }
        /**
         * @brief Performs a masked load of double-precision floating-point values.
         *
         * Loads up to the configured number of active lanes from memory.
         *
         * @param src Source memory.
         *
         * @return Loaded SIMD vector.
         */
        [[nodiscard]]
        __forceinline vec4d load(const double* src) const {
            return vec4d(_mm256_maskload_pd(src, mask4(this->m_size)));
        }
        /**
         * @brief Performs a masked load of 32-bit signed integers.
         *
         * Loads up to the configured number of active lanes from memory.
         *
         * @param src Source memory.
         *
         * @return Loaded SIMD vector.
         */
        [[nodiscard]]
        __forceinline vec8i load(const std::int32_t* src) const {
            return vec8i(_mm256_maskload_epi32(src, mask8(this->m_size)));
        }

        /**
         * @brief Performs a masked store of single-precision floating-point values.
         *
         * Stores only the active lanes into memory.
         *
         * @param dst Destination memory.
         * @param src Source SIMD vector.
         */
        __forceinline void store(float* dst, const vec8f& src) const {
            _mm256_maskstore_ps(dst, mask8(this->m_size), src);
        }
        /**
         * @brief Performs a masked store of double-precision floating-point values.
         *
         * Stores only the active lanes into memory.
         *
         * @param dst Destination memory.
         * @param src Source SIMD vector.
         */
        __forceinline void store(double* dst, const vec4d& src) const {
            _mm256_maskstore_pd(dst, mask4(this->m_size), src);
        }
        /**
         * @brief Performs a masked store of 32-bit signed integers.
         *
         * Stores only the active lanes into memory.
         *
         * @param dst Destination memory.
         * @param src Source SIMD vector.
         */
        __forceinline void store(std::int32_t* dst, const vec8i& src) const {
            _mm256_maskstore_epi32(dst, mask8(this->m_size), src);
        }

        /**
         * @brief Returns the number of active lanes.
         *
         * @return Number of active SIMD lanes represented by this mask.
         */
        [[nodiscard]]
        std::size_t size() const noexcept;
    private:
        std::size_t m_size;
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_MEMORY_MASK_RUNTIME_HPP