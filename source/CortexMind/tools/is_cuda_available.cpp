//
// Created by muham on 15.04.2026.
//

#include "CortexMind/tools/is_cuda_available.hpp"

using namespace cortex;

boolean cortex::has_cuda() {
    #if CXM_IS_CUDA_AVAILABLE
        return true;
    #else //#if CXM_IS_CUDA_AVAILABLE
        return false;
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}