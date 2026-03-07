//
// Created by muham on 21.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CL2_PARAMS_HPP
#define CORTEXMIND_CORE_ENGINE_CL2_PARAMS_HPP

#include <CortexMind/core/Tools/params.hpp>
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

namespace cortex::_fw::cl2 {
    enum class cxm_device : i64 {
        gpu  = CL_DEVICE_TYPE_GPU,
        cpu  = CL_DEVICE_TYPE_CPU,
        any  = CL_DEVICE_TYPE_ALL
    };
} // namespace cortex::_fw::cl2

#endif //CORTEXMIND_CORE_ENGINE_CL2_PARAMS_HPP