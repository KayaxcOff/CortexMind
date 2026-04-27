//
// Created by muham on 27.04.2026.
//

#include "CortexMind/tools/describe.hpp"
#include <CortexMind/framework/Tools/device_as_string.hpp>
#include <CortexMind/framework/Tools/tensor_utils.hpp>
#include <iostream>

using namespace cortex::_fw;
using namespace cortex;

u0 cortex::describe(const tensor &x) {
    std::cout << "Device: " << DeviceAsString(x.device()) << std::endl;
    std::cout << "Elements count: " << x.len() << std::endl;
    std::cout << "Shape : { ";
    for (const auto& item : x.shape()) {
        std::cout << item << " ";
    }
    std::cout << "}" << std::endl;
    std::cout << "Stride: { ";
    for (const auto& item : compute_stride(x.shape())) {
        std::cout << item << " ";
    }
    std::cout << "}" << std::endl;
}