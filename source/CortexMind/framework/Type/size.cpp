//
// Created by muham on 3.08.2026.
//

#include "CortexMind/framework/Type/size.hpp"
#include <tlx/types.hpp>

std::size_t cortex::_fw::size_of(const DType type) noexcept {
    switch (type) {
        case DType::Int32:
            return sizeof(std::int32_t);
        case DType::Int64:
            return sizeof(std::int64_t);
        case DType::Float16:
            return sizeof(half);
        case DType::Float32:
            return sizeof(float);
        case DType::Float64:
            return sizeof(double);
        case DType::BFloat16:
            return sizeof(tlx::bfloat16);
        case DType::QInt16:
            return sizeof(tlx::qint16);
        case DType::QInt8:
            return sizeof(tlx::qint8);
        case DType::QUInt16:
            return sizeof(tlx::quint16);
        case DType::QUInt8:
            return sizeof(tlx::quint8);
        case DType::Unknown:
        default:
            return 0;
    }
}