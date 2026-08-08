//
// Created by muham on 8.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_PARTIAL_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_PARTIAL_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>
#include <cwchar>

namespace cortex::_fw::avx2 {
    struct partial {
        /**
         * @brief Loads a partial AVX2 float vector from memory.
         *
         * Loads the first N elements from the source and sets all remaining
         * lanes to zero.
         *
         * @param src Source array.
         * @param N Number of elements to load. Must be in the range [0, 8].
         * @return AVX2 vector containing the loaded values and zero-filled
         *         remaining lanes.
         */
        [[nodiscard]]
        static vec8f load(const float* src, std::size_t N);
        /**
         * @brief Stores a partial AVX2 float vector to memory.
         *
         * Stores only the first N elements of the vector.
         *
         * @param dst Destination array.
         * @param src Source AVX2 vector.
         * @param N Number of elements to store. Must be in the range [0, 8].
         */
        static void store(float* dst, const vec8f& src, std::size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_PARTIAL_HPP