//
// Created by muham on 3.11.2025.
//

#ifndef CORTEXMIND_RANDOM_WEIGHT_HPP
#define CORTEXMIND_RANDOM_WEIGHT_HPP

#include <random>

namespace cortex::math {
    inline double random_weight(const double min = -0.5, const double max = 0.5) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);
        return dis(gen);
    }
}

#endif //CORTEXMIND_RANDOM_WEIGHT_HPP