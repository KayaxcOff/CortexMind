//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_BIT_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_BIT_HPP

#include <CortexMind/framework/Tools/native.hpp>
#include <tlx/types.hpp>

namespace cortex::_fw {
    namespace avx2 {
        struct vec8f;
    } //namespace avx2

    namespace detail {
        /**
         * @brief Loads sixteen bfloat16 values into two AVX2 float vectors.
         *
         * Converts sixteen consecutive bfloat16 values into two
         * @ref avx2::vec8f vectors suitable for SIMD computation.
         *
         * @param src Pointer to sixteen consecutive bfloat16 values.
         * @param v Destination SIMD vector pair.
         */
        void load_bf16(const tlx::bfloat16* src, native<avx2::vec8f>& v);
        /**
         * @brief Stores two AVX2 float vectors as sixteen bfloat16 values.
         *
         * Converts two @ref avx2::vec8f vectors back into sixteen
         * consecutive bfloat16 values.
         *
         * @param dst Destination buffer.
         * @param v Source SIMD vector pair.
         */
        void store_bf16(tlx::bfloat16* dst, const native<avx2::vec8f>& v);
    } //namespace detail
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_BIT_HPP