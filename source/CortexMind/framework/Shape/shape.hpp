//
// Created by muham on 4.06.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_SHAPE_SHAPE_HPP
#define CORTEXMIND_FRAMEWORK_SHAPE_SHAPE_HPP

#include <CortexMind/framework/Tools/types.hpp>
#include <initializer_list>
#include <vector>

namespace cortex::_fw {
    struct TensorShape {
        TensorShape();
        TensorShape(std::initializer_list<i64> shape);
        explicit TensorShape(const std::vector<i64>& shape);
        TensorShape(const std::vector<i64>& shape, const std::vector<i64>& strides);

        std::vector<i64> shape;
        std::vector<i64> stride;
        i64 offset;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_SHAPE_SHAPE_HPP