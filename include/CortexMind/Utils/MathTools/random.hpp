//
// Created by muham on 9.11.2025.
//

#ifndef CORTEXMIND_RANDOM_HPP
#define CORTEXMIND_RANDOM_HPP

#include <CortexMind/Utils/params.hpp>
#include <random>
#include <vector>

inline std::vector<std::vector<cortex::float32>> random_seed() {
    std::vector<std::vector<cortex::float32>> seed;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dis(0.1f, 1.0f);

    for (cortex::int32 i = 0; i < 3; ++i) {
        std::vector<cortex::float32> row;
        for (cortex::int32 j = 0; j < 3; ++j) {
            row.push_back(dis(gen));
        }
        seed.push_back(row);
    }

    return seed;
}

#endif //CORTEXMIND_RANDOM_HPP