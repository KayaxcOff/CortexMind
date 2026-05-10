//
// Created by muham on 10.05.2026.
//

#include "CortexMind/framework/Tools/as_string.hpp"

std::string cortex::_fw::as_string(const sys::DeviceType d_type) {
    switch (d_type) {
        case sys::DeviceType::HOST: {
            return "host";
        }
        case sys::DeviceType::CUDA: {
            return "cuda";
        }
        default: {
            return "unknown";
        }
    }
}