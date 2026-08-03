//
// Created by muham on 3.08.2026.
//

#include "CortexMind/framework/Type/as_string.hpp"

tlx::vstring cortex::_fw::as_string(const DType type) {
    switch (type) {
        case DType::Int32:
            return "int32";
        case DType::Int64:
            return "int64";
        case DType::Float16:
            return "float16";
        case DType::Float32:
            return "float32";
        case DType::Float64:
            return "float64";
        case DType::BFloat16:
            return "bfloat16";
        case DType::QInt16:
            return "qint16";
        case DType::QInt8:
            return "qint8";
        case DType::QUInt16:
            return "qint16";
        case DType::QUInt8:
            return "qint8";
        case DType::Unknown:
        default:
            return "unknown";
    }
}