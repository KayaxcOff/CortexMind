//
// Created by muham on 25.05.2026.
//

#include "CortexMind/dataset/two_moons.hpp"
#include <numbers>
#include <random>

using namespace cortex::ds;

TwoMoonsDataset::TwoMoonsDataset(const int32 n, float32 noise) : N(n) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution angle(0.0f, std::numbers::pi_v<float32>);
    std::normal_distribution gauss(0.0f, noise);

    this->X.reserve(n * 2);
    this->Y.reserve(n);

    for (int32 i = 0; i < n; ++i) {
        const float32 t = angle(gen);

        if (i < n / 2) {
            const float32 x = std::cos(t);
            const float32 y = std::sin(t);

            this->X.push_back(x + gauss(gen));
            this->X.push_back(y + gauss(gen));
            this->Y.push_back(0.0f);
        } else {
            const float32 x = 1.0f - std::cos(t);
            const float32 y = -std::sin(t) + 0.5f;

            this->X.push_back(x + gauss(gen));
            this->X.push_back(y + gauss(gen));
            this->Y.push_back(1.0f);
        }
    }
}