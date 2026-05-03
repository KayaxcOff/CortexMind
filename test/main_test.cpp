//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    const tensor x1({2, 2});
    const tensor x2({2, 2});

    x1.rand();
    x2.rand();

    std::cout << x1 + x2 << std::endl;

    return 0;
}