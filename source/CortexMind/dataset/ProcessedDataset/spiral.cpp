//
// Created by muham on 25.05.2026.
//

#include "CortexMind/dataset/ProcessedDataset/spiral.hpp"
#include <cmath>
#include <numbers>
#include <random>

using namespace cortex::ds;

Spiral::Spiral(const int32 n, const float32 noise) : N(n) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution gauss(0.0f, noise);

    X.reserve(n * 2);
    Y.reserve(n);

    const int32 half = n / 2;

    for (int32 i = 0; i < half; ++i) {
        const float32 r = static_cast<float32>(i) / static_cast<float32>(half);
        const float32 t = 4.0f * std::numbers::pi_v<float32> * r;

        const float32 x = r * std::cos(t);
        const float32 y = r * std::sin(t);

        this->X.push_back(x + gauss(gen));
        this->X.push_back(y + gauss(gen));
        this->Y.push_back(0.0f);
    }

    for (int i = 0; i < half; ++i) {
        const float32 r = static_cast<float32>(i) / static_cast<float32>(half);
        const float32 t = 4.0f * std::numbers::pi_v<float32> * r + std::numbers::pi_v<float32>;

        const float32 x = r * std::cos(t);
        const float32 y = r * std::sin(t);

        this->X.push_back(x + gauss(gen));
        this->X.push_back(y + gauss(gen));
        this->Y.push_back(1.0f);
    }
}