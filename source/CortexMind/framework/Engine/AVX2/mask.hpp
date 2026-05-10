//
// Created by muham on 7.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>
#include <CortexMind/framework/Tools/types.hpp>
#include <cassert>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2 masked load/store helper (compile-time N).
     *
     * @tparam N Number of elements to operate on (must be in range [1,8])
     */
    template <size_t N>
    struct mask {
        static_assert(N >= 1 && N <= 8, "Mask size N must be in range [1, 8]");
        /**
         * @brief Masked load from memory.
         *
         * @param src Pointer to source array (at least N floats).
         * @return __m256 with first N elements loaded, remaining lanes zero.
         */
        [[nodiscard]]
        static __forceinline vec8f load(const f32* src) {
            return _mm256_maskload_ps(src, get_mask());
        }

        /**
         * @brief Masked store to memory.
         *
         * @param dst Destination pointer.
         * @param src Source vector.
         */
        static __forceinline void store(f32* dst, const vec8f src) {
            _mm256_maskstore_ps(dst, get_mask(), src);
        }
    private:
        /**
         * @brief Returns the precomputed mask at compile time.
         */
        [[nodiscard]]
        static __forceinline vec8i get_mask() {
            alignas(32) static constexpr i32 mask_table[9][8] = {
                {0,  0, 0, 0, 0, 0, 0, 0},   // N=0 (invalid)
                {-1, 0, 0, 0, 0, 0, 0, 0},
                {-1,-1, 0, 0, 0, 0, 0, 0},
                {-1,-1,-1, 0, 0, 0, 0, 0},
                {-1,-1,-1,-1, 0, 0, 0, 0},
                {-1,-1,-1,-1,-1, 0, 0, 0},
                {-1,-1,-1,-1,-1,-1, 0, 0},
                {-1,-1,-1,-1,-1,-1,-1, 0},
                {-1,-1,-1,-1,-1,-1,-1,-1}   // N=8
            };

            return _mm256_load_si256(reinterpret_cast<const vec8i*>(mask_table[N]));
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_HPP