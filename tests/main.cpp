//
// Created by muham on 3.08.2026.
//

#include "CortexMind/framework/Type/type.hpp"
#include <iostream>

using namespace cortex::_fw;

int main() {
    const TensorType type(DType::BFloat16);

    std::cout << type.ToString() << std::endl;

    return 0;
}