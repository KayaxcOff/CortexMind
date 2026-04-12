//
// Created by muham on 8.04.2026.
//

#include "CortexMind/framework/Tools/err.hpp"
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/core/Tools/utils.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/runtime/macros.hpp>
#include <iostream>

using namespace cortex::_fw;

void err::exitIf(const bool status, const std::string &where, const std::string &message) {
    if (!status) {
        std::cerr << "[CXM-Error]: " << where << std::endl;
        std::cerr << message << std::endl;
        std::exit(CXM_ERR_EXIT);
    }
}

#if CXM_IS_CUDA_AVAILABLE
    void err::exitIf(const cudaError_t error, const std::string &where) {
        if (error != cudaSuccess) {
            std::cerr << "[CXM-Error]: " << where << std::endl;
            std::cerr << cuda::ErrorAsString(error) << std::endl;
            std::exit(CXM_ERR_EXIT);
        }
    }
#endif //#if CXM_IS_CUDA_AVAILABLE

void err::warnIf(const bool status, const std::string &where, const std::string &message) {
    if (!status) {
        std::cerr << "[CXM-Error]: " << where << std::endl;
        std::cerr << message << std::endl;
    }
}