//
// Created by muham on 20.04.2026.
//

#include "CortexMind/tools/tensor_info.hpp"
#include <CortexMind/framework/Tools/device_as_string.hpp>
#include <iostream>

using namespace cortex;

u0 cortex::info(const tensor &_tensor) {
    std::cout << "Shape: { ";
    for (const auto &item : _tensor.shape()) {
        std::cout << item << " ";
    }
    std::cout << "}" << std::endl;
    std::cout << "Number of elements: " << _tensor.numel() << std::endl;
    std::cout << "Device: " << _fw::DeviceAsString(_tensor.device()) << std::endl;
}