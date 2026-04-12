//
// Created by muham on 12.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_SOFTMAX_H
#define CORTEXMIND_CORE_ENGINE_CUDA_SOFTMAX_H

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief CUDA Softmax implementation.
     *
     * Internally manages a mapped host/device scalar buffer
     * for max and sum reductions without external dependencies.
     */
    struct Softmax {
        Softmax();
        ~Softmax();

        /**
         * @brief Applies softmax in-place over N elements.
         * @param Xx Device pointer to input/output array
         * @param N  Number of elements
         */
        void forward(f32* Xx, size_t N);

    private:
        f32* host_buf; // [0] = max, [1] = sum
        f32* cuda_buf;
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_SOFTMAX_H