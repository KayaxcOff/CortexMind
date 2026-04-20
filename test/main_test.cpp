//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

int main() {
    const tensor x({3, 3});
    x.rand();

    std::cout << "Tensor X:\n" << x << std::endl;
    std::cout << "Max Value:\n" << x.max() << std::endl;
    std::cout << "Max Value Index:\n" << argmax(x) << std::endl;

    return 0;
}