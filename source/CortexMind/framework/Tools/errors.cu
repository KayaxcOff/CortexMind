//
// Created by muham on 3.08.2026.
//

#include "CortexMind/framework/Tools/errors.hpp"
#include <CortexMind/runtime/macros.hpp>
#include <iostream>
#include <format>

using namespace cortex::_fw;

namespace {
    [[nodiscard]]
    constexpr bool success(const cudaError_t error) {
        return error == cudaSuccess;
    }
} //unnamed namespace

#if CXM_IS_CUDA_AVAILABLE
    void errors::ExitIf(const cudaError_t condition, const tlx::vstring msg, const tlx::vstring file, const int line) {
        if (!success(condition)) {
            std::cerr << std::format(
                "[CUDA ERROR] [{}:{}]\n"
                "  Message : {}\n"
                "  CUDA    : {} ({})\n",
                file,
                line,
                msg,
                cudaGetErrorString(condition),
                static_cast<int>(condition)
            );

            std::exit(CXM_ERR_EXIT);
        }
    }
#endif //#if CXM_IS_CUDA_AVAILABLE