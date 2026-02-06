//
// Created by muham on 2.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX_PARTIAL_HPP
#define CORTEXMIND_CORE_ENGINE_AVX_PARTIAL_HPP

#include <CortexMind/core/Engine/AVX/params.hpp>
#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief Utilities for partial AVX2 load and store operations.
     *
     * This structure provides helper functions to load and store
     * 256-bit AVX vectors (`vec8f`) from/to memory using a logical index.
     * It is intended for scenarios where vectorized access is required
     * but full contiguous alignment cannot be guaranteed.
     */
    struct partial {
        /**
         * @brief Loads a vector of 8 single-precision floats from memory.
         *
         * Reads data starting from the given base pointer and logical index,
         * and returns the result as an AVX2 vector (`vec8f`).
         *
         * @param dest Pointer to the source float array.
         * @param idx  Logical index used to compute the load position.
         * @return Loaded AVX vector containing 8 float values.
         *
         * @note The caller is responsible for ensuring that the memory
         *       region is valid and accessible.
         */
        static vec8f load(const f32* dest, size_t idx);

        /**
         * @brief Stores a vector of 8 single-precision floats into memory.
         *
         * Writes the contents of the given AVX2 vector (`vec8f`) into
         * the destination memory location determined by the base pointer
         * and logical index.
         *
         * @param dest Pointer to the destination float array.
         * @param val  AVX vector containing the values to store.
         * @param idx  Logical index used to compute the store position.
         *
         * @note The caller must ensure that the destination memory
         *       region is writable.
         */
        static void store(f32* dest, vec8f val, size_t idx);
    };
} // namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX_PARTIAL_HPP