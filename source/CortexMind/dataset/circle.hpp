//
// Created by muham on 23.05.2026.
//

#ifndef CORTEXMIND_DATASET_CIRCLE_HPP
#define CORTEXMIND_DATASET_CIRCLE_HPP

#include <CortexMind/tools/types.hpp>
#include <vector>

namespace cortex::ds {
    /**
     * @brief Synthetic 2D circle dataset for binary classification.
     *
     * Generates points distributed in a 2D plane with two classes:
     * - Class 1: Points inside the unit circle (radius < 0.5)
     * - Class 0: Points outside the unit circle
     *
     * Gaussian noise can be added to make the problem more challenging.
     */
    struct CircleDataset {
        std::vector<float32> X;   ///< Flattened input features: [x0, y0, x1, y1, ...]
        std::vector<float32> Y;   ///< Labels: 0 or 1

        int32 N;

        /**
         * @brief Constructs a Circle dataset.
         *
         * @param n     Number of samples to generate
         * @param noise Standard deviation of Gaussian noise added to points (default: 0.1)
         */
        explicit CircleDataset(int32 n, float32 noise = 0.1f);
    };
} //namespace cortex::ds

#endif //CORTEXMIND_DATASET_CIRCLE_HPP