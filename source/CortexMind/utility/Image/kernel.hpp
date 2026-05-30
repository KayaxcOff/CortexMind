//
// Created by muham on 30.05.2026.
//

#ifndef CORTEXMIND_UTILITY_IMAGE_KERNEL_HPP
#define CORTEXMIND_UTILITY_IMAGE_KERNEL_HPP

#include <CortexMind/tools/types.hpp>

namespace cortex::utils {
    class SpatialKernel {
    public:
        [[nodiscard]]
        static tensor apply(const tensor& Xx, const tensor& Xy);
    };
} //namespace cortex::utils

#endif //CORTEXMIND_UTILITY_IMAGE_KERNEL_HPP