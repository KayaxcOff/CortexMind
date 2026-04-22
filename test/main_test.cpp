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

    const auto z = x + y;
    z.backward();

    std::cout << "Tensor Z:\n" << z << std::endl;
    std::cout << "Gradient X:\n" << x.grad() << std::endl;
    std::cout << "Gradient Y:\n" << y.grad() << std::endl;

    return 0;
}