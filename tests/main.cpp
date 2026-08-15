//
// Created by muham on 3.08.2026.
//

#include <CortexMind/framework/Shape/shape.hpp>
#include <iostream>

using namespace cortex::_fw;

int main() {
    TensorShape shape({2, 2});

    std::cout << "Shape: {";
    for (const auto& item : shape.shape()) {
        std::cout << item;
    }
    std::cout << "}" << std::endl;

    std::cout << "Stride: {";
    for (const auto& item : shape.stride()) {
        std::cout << item;
    }
    std::cout << "}" << std::endl;

    return 0;
}