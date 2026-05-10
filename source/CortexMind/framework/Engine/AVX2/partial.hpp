//
// Created by muham on 7.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_PARTIAL_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_PARTIAL_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>
#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief Partial load and store operations for __m256 vectors.
     *
     * This struct contains scalar-based helper functions to handle
     * boundary conditions where the number of elements is not a multiple of 8.
     */
    struct partial {
        /**
         * @brief Loads the first N elements from memory into a __m256 vector.
         *
         * Remaining lanes (from N to 7) are set to zero.
         *
         * @param src Pointer to source memory.
         * @param N   Number of elements to load (0 ≤ N ≤ 8).
         * @return __m256 vector with first N elements loaded, rest zeroed.
         *
         * @note This is a scalar fallback implementation (loop-based).
         *       For higher performance, consider using `mask<N>::load()` instead.
         *
         * @warning No bounds checking is performed on `src`. Caller must ensure
         *          at least N floats are readable.
         */
        [[nodiscard]]
        static vec8f load(const f32* src, size_t N);
        /**
         * @brief Stores the first N elements from a vec8f vector to memory.
         *
         * Only the first N lanes of the vector are written to `dst`.
         *
         * @param dst Destination memory pointer.
         * @param src Source vec8f vector.
         * @param N   Number of elements to store (0 ≤ N ≤ 8).
         *
         * @note This is a scalar fallback implementation (loop-based).
         *       For higher performance, consider using `mask<N>::store()` instead.
         *
         * @warning No bounds checking is performed on `dst`. Caller must ensure
         *          at least N floats are writable.
         */
        static void store(f32* dst, vec8f src, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_PARTIAL_HPP