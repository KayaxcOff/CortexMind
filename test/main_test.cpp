//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    const tensor x({2, 2});
    x.uniform(1, 3);

    std::cout << x << std::endl;

    return 0;
}

/**
 * output:
 * [[2.77251, 2.26996],
 *  [2.13977, 2.52902]]
 */