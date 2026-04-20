//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    const tensor x({3, 5, 6});
    x.rand();

    std::cout << x.sum() << std::endl;

    return 0;
}