//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_CAST_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_CAST_HPP

#include <tlx/types.hpp>

namespace cortex::_fw {
    /**
     * @brief Converts bfloat16 values into single-precision floating-point values.
     *
     * Converts @p size consecutive bfloat16 elements into IEEE-754
     * single-precision floating-point values.
     *
     * Implementations may utilize SIMD instructions when available
     * and automatically fall back to scalar conversion for remaining
     * elements.
     *
     * @param dst Destination float buffer.
     * @param src Source bfloat16 buffer.
     * @param size Number of elements to convert.
     */
    void convert(float* dst, const tlx::bfloat16* src, std::size_t size);
    /**
     * @brief Converts IEEE FP16 values into single-precision floating-point values.
     *
     * Converts @p size consecutive half-precision floating-point values
     * into IEEE-754 single-precision floating-point values.
     *
     * Implementations may utilize SIMD instructions (such as F16C)
     * when available and automatically fall back to scalar conversion
     * for remaining elements.
     *
     * @param dst Destination float buffer.
     * @param src Source FP16 buffer.
     * @param size Number of elements to convert.
     */
    void convert(float* dst, const tlx::half* src, std::size_t size);
    /**
     * @brief Converts single-precision floating-point values into bfloat16 values.
     *
     * Converts @p size consecutive float values into bfloat16 values
     * using round-to-nearest-even semantics whenever SIMD conversion
     * is available.
     *
     * @param dst Destination bfloat16 buffer.
     * @param src Source float buffer.
     * @param size Number of elements to convert.
     */
    void convert(tlx::bfloat16* dst, const float* src, std::size_t size);
    /**
     * @brief Converts single-precision floating-point values into IEEE FP16 values.
     *
     * Converts @p size consecutive float values into half-precision
     * floating-point values.
     *
     * Implementations may utilize SIMD instructions (such as F16C)
     * when available and automatically fall back to scalar conversion
     * for remaining elements.
     *
     * @param dst Destination FP16 buffer.
     * @param src Source float buffer.
     * @param size Number of elements to convert.
     */
    void convert(tlx::half* dst, const float* src, std::size_t size);
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_CAST_HPP