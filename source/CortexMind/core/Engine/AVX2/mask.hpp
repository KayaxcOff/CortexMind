//
// Created by muham on 7.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_MASK_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_MASK_HPP

#include <CortexMind/core/Engine/AVX2/params.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2 masked load/store helper for partial (boundary) vector operations.
     *
     * This struct creates a 256-bit mask based on the number of valid elements (N ≤ 8)
     * and provides masked load and store operations using `_mm256_maskload_ps` and
     * `_mm256_maskstore_ps`. It is primarily used when tensor lengths are not multiples of 8.
     */
    struct mask {
        /**
         * @brief Constructs a mask for the first N elements.
         * @param N Number of valid elements (0 ≤ N ≤ 8)
         */
        explicit mask(const size_t N) : N(N) {
            this->m_mask = this->create();
        }

        /**
         * @brief Performs a masked load of the first N elements from memory.
         *
         * Elements beyond N are set to zero in the returned vector.
         *
         * @param src Pointer to source memory
         * @return __m256 containing the first N elements, remaining lanes zeroed
         */
        [[nodiscard]]
        __forceinline vec8f load(const f32* src) const {
            return _mm256_maskload_ps(src, this->m_mask);
        }
        /**
         * @brief Performs a masked store of the first N elements to memory.
         *
         * Only the first N lanes are written to the destination; remaining elements are untouched.
         *
         * @param dst Destination pointer
         * @param src Source vector
         */
        __forceinline void store(f32* dst, const vec8f src) const {
            _mm256_maskstore_ps(dst, this->m_mask, src);
        }
    private:
        size_t N;
        vec8i m_mask{};

        /**
         * @brief Creates the AVX2 mask integer vector for the first N elements.
         *
         * Uses a pre-defined lookup table (`mask_data`) and `_mm256_loadu_si256`
         * to generate the correct 256-bit mask.
         */
        [[nodiscard]]
        __forceinline vec8i create() const {
            alignas(32) static constexpr int32_t mask_data[16] = {
                -1, -1, -1, -1, -1, -1, -1, -1,
                 0,  0,  0,  0,  0,  0,  0,  0
            };
            return _mm256_loadu_si256(reinterpret_cast<const vec8i*>(&mask_data[8 - this->N]));
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_MASK_HPP