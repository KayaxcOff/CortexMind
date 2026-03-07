//
// Created by muham on 22.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_MEMORY_DEVICE_HPP
#define CORTEXMIND_CORE_ENGINE_MEMORY_DEVICE_HPP

#define CXM_DEVICE_HOST 0
#define CXM_DEVICE_CUDA 1

#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::sys {
    enum class device : i32 {
        host = CXM_DEVICE_HOST,
        cuda = CXM_DEVICE_CUDA,
    };
} // namespace cortex::_fw::sys

#endif //CORTEXMIND_CORE_ENGINE_MEMORY_DEVICE_HPP