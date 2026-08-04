//
// Created by muham on 3.08.2026.
//

#include "CortexMind/framework/Tools/errors.hpp"
#include <CortexMind/framework/Tools/Log/as_string.hpp>
#include <CortexMind/framework/Tools/Log/w.hpp>
#include <CortexMind/framework/Tools/console.hpp>
#include <CortexMind/runtime/macros.hpp>
#include <iostream>

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
            CXM_INITIALIZE_CONSOLE();
            constexpr auto level = LogLevel::ERROR;
            WLog(level)
            << "[" << as_string(level) << "]" << "\n"
            << "[" << file << ":" << line << "] | " << msg << std::endl;
            std::exit(CXM_ERR_EXIT);
        }
    }
#endif //#if CXM_IS_CUDA_AVAILABLE