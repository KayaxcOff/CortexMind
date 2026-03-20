//
// Created by muham on 13.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_PARTIAL_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_PARTIAL_HPP

#include <CortexMind/core/Engine/AVX2/params.hpp>
#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief Utility for partial (prefix) vector load and store operations
     *
     * Contains static methods to load/store only the first `n` elements of a `vec8f`,
     * zero-filling or ignoring the remaining lanes.
     *
     * Intended for handling loop remainders in AVX2-only code paths where masking
     * is not available or not desired.
     */
    struct partial {
        /**
         * @brief   Loads the first `idx` floats from memory, zero-fills the rest
         * @param   dest  Pointer to source memory (must be readable for ≥ `idx` elements)
         * @param   idx   How many elements to load (valid range: 0 ≤ idx ≤ 8)
         * @return  vec8f with lanes [0.idx-1] filled from memory, [idx..7] = 0.0f
         *
         * @pre idx ≤ 8
         * @pre dest is valid for at least `idx` contiguous floats
         */
        [[nodiscard]]
        static vec8f load(const f32* dest, size_t idx);
        /**
         * @brief   Stores the first `idx` lanes of the vector to memory
         * @param   dest  Pointer to destination memory (must be writable for ≥ `idx` elements)
         * @param   val   Vector whose prefix will be stored
         * @param   idx   How many elements to write (valid range: 0 ≤ idx ≤ 8)
         *
         * @pre idx ≤ 8
         * @pre dest is writable for at least `idx` contiguous floats
         *
         * @note    Does **not** modify memory beyond index `idx-1`
         */
        static void store(f32* dest, vec8f val, size_t idx);
    };
} // namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_PARTIAL_HPP