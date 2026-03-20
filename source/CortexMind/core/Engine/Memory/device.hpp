//
// Created by muham on 14.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_MEMORY_DEVICE_HPP
#define CORTEXMIND_CORE_ENGINE_MEMORY_DEVICE_HPP

#include <CortexMind/core/Tools/defaults.hpp>
#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::sys {
    /**
     * @brief Class that specifies CPU
     * and GPU drivers
     */
    enum class dev : i64 {
        host = CXM_HOST_DEVICE,     ///< CPU Driver
        cuda = CXM_CUDA_DEVICE,     ///< GPU Driver
    };
} // namespace cortex::_fw::sys

#endif //CORTEXMIND_CORE_ENGINE_MEMORY_DEVICE_HPP