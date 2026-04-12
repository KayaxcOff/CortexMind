//
// Created by muham on 12.04.2026.
//

#include "CortexMind/framework/Tools/device_as_string.hpp"

using namespace cortex::_fw::sys;

std::string_view cortex::_fw::sys::DeviceAsString(const deviceType d_type) {
    switch (d_type) {
        case host: return "host";
        case cuda: return "cuda";
        default: return "unknown";
    }
}