//
// Created by muham on 13.08.2026.
//

#ifndef CORTEXMIND_TOOLS_TENSOR_INFO_HPP
#define CORTEXMIND_TOOLS_TENSOR_INFO_HPP

#include <CortexMind/framework/Memory/type.hpp>
#include <CortexMind/framework/Type/type.hpp>
#include <CortexMind/runtime/macros.hpp>
#include <tlx/vector.hpp>

namespace cortex {
    struct TensorInfo {
        tlx::vec<std::int64_t, CXM_MAX_DIMS> _shape;
        _fw::DType _dtype = _fw::DType::Float32;
        _fw::DeviceType _deviceType = _fw::DeviceType::HOST;
        bool _requires_grad = false;
    };
} //namespace cortex

#endif //CORTEXMIND_TOOLS_TENSOR_INFO_HPP