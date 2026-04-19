//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    tensor x({3, 5, 6}, host);
    const tensor y({3, 5, 6}, host);

    x.rand();
    y.rand();

    std::cout << "Tensor X:\n" << x << std::endl;
    std::cout << "Tensor Y:\n" << y << std::endl;

    x += y;

    std::cout << "Tensor Z:\n" << x << std::endl;

    return 0;
}