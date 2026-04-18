//
// Created by muham on 12.04.2026.
//

#include "CortexMind/framework/Tools/device_as_string.hpp"

std::string cortex::_fw::DeviceAsString(const sys::deviceType d_type) {
    switch (d_type) {
        case sys::host: return "host";
        case sys::cuda: return "cuda";
        default: return "unknown";
    }
}