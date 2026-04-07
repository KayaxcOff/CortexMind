//
// Created by muham on 7.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_PARTIAL_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_PARTIAL_HPP

#include <CortexMind/core/Engine/AVX2/params.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief Partial (masked) load and store operations for AVX2 vectors.
     *
     * Provides functions to load and store only the first N elements of a __m256 vector,
     * useful for handling tensor operations with lengths not divisible by 8.
     */
    struct partial {
        /**
         * @brief Loads the first N elements from memory into a __m256 vector.
         *
         * Remaining elements (beyond N) are set to zero.
         *
         * @param src Pointer to the source memory
         * @param N   Number of elements to load (0 ≤ N ≤ 8)
         * @return __m256 with first N elements loaded, rest zeroed
         */
        [[nodiscard]]
        static vec8f load(const f32* src, size_t N);
        /**
         * @brief Stores the first N elements of a __m256 vector to memory.
         *
         * Only the first N lanes are written to destination.
         *
         * @param dst Pointer to the destination memory
         * @param src Source vector
         * @param N   Number of elements to store (0 ≤ N ≤ 8)
         */
        static void store(f32* dst, vec8f src, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_PARTIAL_HPP