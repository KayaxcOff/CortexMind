//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    const tensor x({3, 3, 3});
    constexpr float32 x_scale = 1.0f;

    x.rand();

    const auto y = x + x_scale;

    std::cout << "Tensor X:\n" << x << std::endl;
    std::cout << "Scalar value:\n" << x_scale << std::endl;
    std::cout << "Addition between X and Scaler value:\n" << y << std::endl;

    return 0;
}