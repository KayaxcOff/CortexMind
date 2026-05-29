//
// Created by muham on 10.05.2026.
//

#include "CortexMind/framework/Tools/as_string.hpp"

std::string cortex::_fw::as_string(const sys::DeviceType d_type) {
    switch (d_type) {
        case sys::DeviceType::kHOST: {
            return "host";
        }
        case sys::DeviceType::kCUDA: {
            return "cuda";
        }
        default: {
            return "unknown";
        }
    }
}

std::string cortex::_fw::as_string(const BroadcastKind kind) {
    switch (kind) {
        case BroadcastKind::kCol: {
            return "col";
        }
        case BroadcastKind::kRow: {
            return "row";
        }
        case BroadcastKind::kNone: {
            return "none";
        }
        case BroadcastKind::kGeneral: {
            return "general";
        }
        default: {
            return "unknown";
        }
    }
}

std::string cortex::_fw::as_string(const DType dtype) {
    switch (dtype) {
        case DType::Bool: {
            return "bool";
        }
        case DType::Float32: {
            return "float32";
        }
        case DType::String: {
            return "string";
        }
        default: {
            return "unknown";
        }
    }
}