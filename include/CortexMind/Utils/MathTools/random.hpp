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

inline std::vector<cortex::tensor> random_weights(const cortex::size in, const cortex::size out) {
    std::vector<cortex::tensor> weights;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dis(-1.0f, 1.0f);

    cortex::tensor weight_row{{}, {}};

    for (cortex::size i = 0; i < in; ++i) {

        for (cortex::size j = 0; j < out; ++j) {
            weight_row = cortex::tensor{{static_cast<size_t>(dis(gen))}, {}};
        }
        weights.push_back(weight_row);
    }

    return weights;
}

#endif //CORTEXMIND_RANDOM_HPP