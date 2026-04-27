//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    const tensor x({2, 2});
    x.rand(2, 5);

    std::cout << "Tensor\n" << x << std::endl;
    std::cout << "--- Tensor Info ----" << std::endl;
    describe(x);

    return 0;
}