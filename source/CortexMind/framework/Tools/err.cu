//
// Created by muham on 8.04.2026.
//

#include "CortexMind/framework/Tools/err.hpp"
#include <CortexMind/runtime/macros.hpp>
#include <iostream>

using namespace cortex::_fw;

void err::exitIf(cudaError_t error, const std::string& where) {
    if(error != cudaSuccess) {
        std::cerr << "[CXM-CUDA-Error]: " << where << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
        std::exit(CXM_ERR_EXIT);
    }
}