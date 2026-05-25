//
// Created by muham on 23.05.2026.
//

#include "CortexMind/dataset/ProcessedDataset/circle.hpp"
#include <cmath>
#include <random>

using namespace cortex::ds;

Circle::Circle(const int32 n, const float32 noise) : N(n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(-1.0f, 1.0f);
    std::normal_distribution gauss(0.0f, noise);

    this->X.reserve(n * 2);
    this->Y.reserve(n);

    for (int32 i = 0; i < n; i++) {
        float32 x = dist(gen);
        float32 y = dist(gen);

        const float32 r = std::sqrt(x*x + y*y);

        float32 label = (r < 0.5f) ? 1.0f : 0.0f;

        x += gauss(gen);
        y += gauss(gen);

        this->X.push_back(x);
        this->X.push_back(y);
        this->Y.push_back(label);
    }
}