//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    utils::TensorFactory factory;
    factory.Set(1.0f);
    std::cout << factory.as_tensor() << std::endl;

    return 0;
}