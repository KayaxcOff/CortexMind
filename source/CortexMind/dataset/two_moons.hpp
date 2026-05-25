//
// Created by muham on 25.05.2026.
//

#ifndef CORTEXMIND_DATASET_TWO_MOONS_HPP
#define CORTEXMIND_DATASET_TWO_MOONS_HPP

#include <CortexMind/tools/types.hpp>
#include <vector>

namespace cortex::ds {
    struct TwoMoonsDataset {
        int32 N;
        std::vector<float32> X;
        std::vector<float32> Y;

        explicit TwoMoonsDataset(int32 n, float32 noise = 0.05f);
    };
} //namespace cortex::ds

#endif //CORTEXMIND_DATASET_TWO_MOONS_HPP