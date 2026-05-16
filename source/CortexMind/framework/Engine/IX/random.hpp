//
// Created by muham on 16.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_RANDOM_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_RANDOM_HPP

#include <CortexMind/framework/Storage/stor.hpp>

namespace cortex::_fw::ix {
    /**
     * @brief Static dispatch for random number generation.
     *
     * CPU path: std::mt19937 with thread_local state.
     * CUDA path: cuRAND via runtime::Curand singleton.
     */
    struct RandomOp {
        /**
         * @brief Fills storage with uniform random values in [min, max].
         *
         * @param x   Target storage
         * @param min Lower bound
         * @param max Upper bound
         * @param N   Number of elements
         */
        static void uniform(TensorStorage* x, f32 min, f32 max, size_t N);
        /**
         * @brief Fills storage with normally distributed random values.
         *
         * @param x    Target storage
         * @param mean Distribution mean
         * @param std  Standard deviation
         * @param N    Number of elements
         */
        static void normal(TensorStorage* x, f32 mean, f32 std, size_t N);
    };
} //namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_RANDOM_HPP