//
// Created by muham on 27.04.2026.
//

#ifndef CORTEXMIND_TOOLS_DESCRIBE_HPP
#define CORTEXMIND_TOOLS_DESCRIBE_HPP

#include <CortexMind/tools/params.hpp>

namespace cortex {
    /**
     * @brief Writes device, total element, shape and stride of tensor
     */
    u0 describe(const tensor& x);
} //namespace cortex

#endif //CORTEXMIND_TOOLS_DESCRIBE_HPP