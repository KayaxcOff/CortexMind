//
// Created by muham on 12.04.2026.
//

#include "CortexMind/tools/is_cuda_available.hpp"
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/core/Tools/utils.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/err.hpp>
#include <iostream>

using namespace cortex::_fw;

void cortex::is_cuda_available() {
    int device;

    CXM_CUDA_ASSERT(cuda::GetDevice(&device), "cortex::is_cuda_available()");

    if (device == 0) {
        std::cout << "Can't found CUDA" << std::endl;
    } else {
        std::cout << "Found CUDA! GPU Element: " << device << std::endl;
    }
}
