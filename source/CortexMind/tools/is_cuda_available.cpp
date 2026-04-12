//
// Created by muham on 12.04.2026.
//

#include "CortexMind/tools/is_cuda_available.hpp"

using namespace cortex;

bool cortex::is_cuda_available() noexcept {
    #if CXM_IS_CUDA_AVAILABLE
        return true;
    #else //#if CXM_IS_CUDA_AVAILABLE
        return false;
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}
