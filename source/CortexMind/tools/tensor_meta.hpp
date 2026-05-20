//
// Created by muham on 20.05.2026.
//

#ifndef CORTEXMIND_TOOLS_TENSOR_META_HPP
#define CORTEXMIND_TOOLS_TENSOR_META_HPP

#include <CortexMind/tools/types.hpp>

namespace cortex {
    /**
     * @brief Returns index of max value of tensor
     * @param x Tensor
     * @return Index
     */
    int32 argmax(const tensor& x);
    /**
     * @brief Returns index of min value of tensor
     * @param x Tensor
     * @return Index
     */
    int32 argmin(const tensor& x);
} //namespace cortex

#endif //CORTEXMIND_TOOLS_TENSOR_META_HPP