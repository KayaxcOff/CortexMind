//
// Created by muham on 16.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_FILL_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_FILL_HPP

#include <CortexMind/framework/Storage/stor.hpp>

namespace cortex::_fw::ix {
    /**
     * @brief Static dispatch for fill, zero, and ones operations.
     *
     * Routes to AVX2 (HOST) or CUDA backend based on storage device.
     * Uses vectorized paths for performance.
     */
    struct FillOp {
        /**
         * @brief Fills all elements with a scalar value.
         * @param x     Target storage
         * @param value Fill value
         * @param N     Number of elements
         */
        static void fill(TensorStorage* x, f32 value, size_t N);
        /**
         * @brief Sets all elements to 0.0f.
         */
        static void zero(TensorStorage* x, size_t N);
        /**
         * @brief Sets all elements to 1.0f.
         */
        static void ones(TensorStorage* x, size_t N);
    };
} //namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_FILL_HPP