//
// Created by muham on 3.08.2026.
//

#include "as_string.hpp"

tlx::vstring cortex::_fw::as_string(const DeviceType type) {
    switch (type) {
        case DeviceType::HOST:
            return "cpu";
        case DeviceType::CUDA:
            return "cuda";
        default:
            return "unknown";
    }
}