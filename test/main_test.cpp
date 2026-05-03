//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    nn::Dense dense(2, 2, host);
    tensor input({1, 2});
    input.rand();

    std::cout << dense.forward(input) << std::endl;

    return 0;
}

/*
 * output:
 * [[0.291248, 0.0591238]]
 */