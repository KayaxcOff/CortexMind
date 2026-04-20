//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    tensor x({3, 3}, cuda);
    tensor y({3, 3}, cuda);
    x.rand();
    y.rand();

    const auto z = x.dot(y);

    std::cout << "Tensor X:\n" << x << std::endl;
    std::cout << "Tensor Y:\n" << y << std::endl;
    std::cout << "Tensor Z:\n" << z << std::endl;

    return 0;
}