//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    tensor x({2, 2}, host, true);
    x.rand();

    std::cout << "Tensor X\n" << x << std::endl;

    tensor y({2, 2}, host, true);
    y.rand();

    std::cout << "Tensor Y\n" << y << std::endl;

    const tensor z = x + y;
    std::cout << "Tensor Z\n" << z << std::endl;

    z.sum().backward();

    std::cout << "Gradient X\n" << x.grad() << std::endl;
    std::cout << "Gradient Y\n" << y.grad() << std::endl;

    return 0;
}