//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

int main() {
    tensor x({2, 2}, host, true);
    tensor y({2, 2}, host, true);

    x.rand();
    y.rand();

    const auto z = x * y;
    z.backward();

    std::cout << "Tensor Z:\n" << z << std::endl;
    std::cout << "Gradient X:\n" << x.grad() << std::endl;
    std::cout << "Gradient Y:\n" << y.grad() << std::endl;

    return 0;
}

/*
 * --- output ---
 * Tensor Z:
 * [[0.128092, 0.367961],
 * [0.120822, 0.0935796]]

 * Gradient X:
 * [[0, 0],
 * [0, 0]]

 * Gradient Y:
 * [[0, 0],
 * [0, 0]]
 */