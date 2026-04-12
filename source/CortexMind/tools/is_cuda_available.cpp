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
    #if CXM_IS_CUDA_AVAILABLE
        int device;
        CXM_CUDA_ASSERT(cuda::GetDevice(&device), "cortex::is_cuda_available()");
        std::cout << "CUDA available, device: " << device << std::endl;
    #else //#if CXM_IS_CUDA_AVAILABLE
        std::cout << "CUDA not compiled." << std::endl;
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}
