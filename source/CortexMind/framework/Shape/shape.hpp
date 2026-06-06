//
// Created by muham on 4.06.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_SHAPE_SHAPE_HPP
#define CORTEXMIND_FRAMEWORK_SHAPE_SHAPE_HPP

#include <CortexMind/framework/Tools/types.hpp>
#include <CortexMind/runtime/macros.hpp>
#include <array>
#include <initializer_list>
#include <span>

namespace cortex::_fw {
    struct TensorShape {
        TensorShape();
        TensorShape(std::initializer_list<i64> _shape);
        explicit TensorShape(const std::span<const i64>& _shape);

        std::array<i64, CXM_MAX_DIMS> shape;
        std::array<i64, CXM_MAX_DIMS> stride;
        i64 offset;
        i32 ndim;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_SHAPE_SHAPE_HPP