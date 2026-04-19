//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    auto x = tensor({2, 2}, host, true);
    x.rand();

    std::cout << "Tensor:\n" << x << std::endl;
    std::cout << "Gradient:\n" << x.grad() << std::endl;

    return 0;
}