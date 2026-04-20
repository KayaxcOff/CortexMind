//
// Created by muham on 20.04.2026.
//

#ifndef CORTEXMIND_RUNTIME_PROVIDER_CUH
#define CORTEXMIND_RUNTIME_PROVIDER_CUH

#include <CortexMind/framework/Tools/params.hpp>
#include <cublas_v2.h>

namespace cortex::_fw::runtime {
    struct Provider {
        cublasHandle_t handle;

        static Provider& instance();
    private:
        Provider(i32 device_id = 0);
        ~Provider();
    };
} //namespace cortex::_fw::runtime

#endif //CORTEXMIND_RUNTIME_PROVIDER_CUH