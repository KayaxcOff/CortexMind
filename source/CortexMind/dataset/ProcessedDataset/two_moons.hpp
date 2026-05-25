//
// Created by muham on 25.05.2026.
//

#ifndef CORTEXMIND_DATASET_PROCESSED_DATASET_TWO_MOONS_HPP
#define CORTEXMIND_DATASET_PROCESSED_DATASET_TWO_MOONS_HPP

#include <CortexMind/tools/types.hpp>
#include <vector>

namespace cortex::ds {
    /**
     * @brief Two Moons (Two Half-Moons) synthetic dataset.
     *
     * Generates a classic binary classification dataset consisting of two interleaving
     * half-moon shapes. This is a non-linearly separable dataset commonly used to test
     * the capability of neural networks to learn non-linear decision boundaries.
     */
    struct TwoMoons {
        int32 N;                      ///< Number of samples
        std::vector<float32> X;       ///< Input features (flattened): [x0, y0, x1, y1, ...]
        std::vector<float32> Y;       ///< Labels: 0 or 1

        /**
         * @brief Constructs a Two Moons dataset.
         *
         * @param n     Number of samples to generate
         * @param noise Standard deviation of Gaussian noise added to points (default: 0.05)
         */
        explicit TwoMoons(int32 n, float32 noise = 0.05f);
    };
} //namespace cortex::ds

#endif //CORTEXMIND_DATASET_PROCESSED_DATASET_TWO_MOONS_HPP