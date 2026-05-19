//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    const tensor x1({2, 2}, host, true);
    const tensor x2({2, 2}, host, true);

    x1.uniform();
    x2.uniform();

    const auto z = x1 * x2;

    std::cout << "Tensor 1:\n" << x1 << std::endl;
    std::cout << "Tensor 2:\n" << x2 << std::endl;
    std::cout << "Result:\n" << z << std::endl;

    z.sum().backward();

    std::cout << "Gradient 1:\n" << x1.grad() << std::endl;
    std::cout << "Gradient 2:\n" << x2.grad() << std::endl;

    return 0;
}

/*
Tensor 1:
[[0.40486, 0.904885],
 [0.0911795, 0.791511]]
Tensor 2:
[[0.929248, 0.137958],
 [0.800918, 0.150254]]
Result:
[[1.33411, 1.04284],
 [0.892097, 0.941765]]
Gradient 1:
[[0, 0],
 [0, 0]]
Gradient 2:
[[0, 0],
 [0, 0]]
*/