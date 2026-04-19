//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    const utils::TensorFactory factory(2.0f);

    std::cout << factory.cast() << std::endl;

    return 0;
}