//
// Created by muham on 22.05.2026.
//

#ifndef CORTEXMIND_TOOLS_MATH_HPP
#define CORTEXMIND_TOOLS_MATH_HPP

#include <CortexMind/tools/types.hpp>

namespace cortex {
    /**
     * @brief Z = X + Y
     * @param Xx First input tensor
     * @param Xy Second input tensor
     * @return Result
     */
    tensor add(const tensor& Xx, const tensor& Xy);
    /**
     * @brief Z = X - Y
     * @param Xx First input tensor
     * @param Xy Second input tensor
     * @return Result
     */
    tensor sub(const tensor& Xx, const tensor& Xy);
    /**
     * @brief Z = X + Y
     * @param Xx First input tensor
     * @param Xy Second input tensor
     * @return Result
     */
    tensor mul(const tensor& Xx, const tensor& Xy);
    /**
     * @brief Z = X / Y
     * @param Xx First input tensor
     * @param Xy Second input tensor
     * @return Result
     */
    tensor div(const tensor& Xx, const tensor& Xy);
    /**
     * @brief Z = X @ Y
     * @param Xx First input tensor
     * @param Xy Second input tensor
     * @return Result
     */
    tensor matmul(const tensor& Xx, const tensor& Xy);
} //namespace cortex

#endif //CORTEXMIND_TOOLS_MATH_HPP