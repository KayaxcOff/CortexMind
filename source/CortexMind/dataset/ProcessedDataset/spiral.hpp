//
// Created by muham on 25.05.2026.
//

#ifndef CORTEXMIND_DATASET_PROCESSED_DATASET_SPIRAL_HPP
#define CORTEXMIND_DATASET_PROCESSED_DATASET_SPIRAL_HPP

#include <CortexMind/tools/types.hpp>
#include <vector>

namespace cortex::ds {
    /**
     * @brief Two intertwined spirals synthetic dataset for binary classification.
     *
     * Generates a challenging non-linearly separable dataset consisting of two
     * spiral arms. This dataset is commonly used to test a model's ability to
     * learn complex decision boundaries.
     */
    struct Spiral {
        int32 N;                      ///< Number of samples
        std::vector<float32> X;       ///< Input features (flattened): [x0, y0, x1, y1, ...]
        std::vector<float32> Y;       ///< Labels: 0 or 1

        /**
         * @brief Constructs a Spiral dataset.
         *
         * @param n     Number of samples to generate
         * @param noise Standard deviation of Gaussian noise added to points (default: 0.05)
         */
        explicit Spiral(int32 n, float32 noise = 0.05f);
    };
} //namespace cortex::ds

#endif //CORTEXMIND_DATASET_PROCESSED_DATASET_SPIRAL_HPP