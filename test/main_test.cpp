//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    const tensor x1({2, 2});
    const tensor x2({2, 2});

    x1.uniform();
    x2.uniform();

    std::cout << "Tensor 1:\n" << x1 << std::endl;
    std::cout << "Tensor 2:\n" << x2 << std::endl;
    std::cout << "Result:\n" << x1 + x2 << std::endl;

    return 0;
}

/*
 * Tensor 1:
 * [[0.575094, 0.447581],
 *  [0.743612, 0.741169]]
 * Tensor 2:
 * [[0.871806, 0.74327],
 *  [0.993726, 0.226882]]
 * Result:
 * [[1.4469, 1.19085],
 *  [1.73734, 0.968051]]
 */