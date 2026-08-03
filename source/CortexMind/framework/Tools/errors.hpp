//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_ERRORS_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_ERRORS_HPP

#include <tlx/string.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <cuda_runtime.h>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <source_location>

namespace cortex::_fw::errors {
    void ExitIf(bool condition, tlx::vstring msg, const std::source_location& sloc = std::source_location::current());
    void WarnIf(bool condition, tlx::vstring msg, const std::source_location& sloc = std::source_location::current());
    #if CXM_IS_CUDA_AVAILABLE
        void ExitIf(cudaError_t condition, tlx::vstring msg, tlx::vstring file, int line);
    #endif //#if CXM_IS_CUDA_AVAILABLE
} //namespace cortex::_fw::errors

#define CXM_ASSERT(cond, msg) \
    ::cortex::_fw::errors::ExitIf((cond), (msg))

#define CXM_WARN(cond, msg) \
    ::cortex::_fw::errors::WarnIf((cond), (msg))

#if CXM_IS_CUDA_AVAILABLE
    #define CXM_DEVICE_ASSERT(cond, msg) \
        ::cortex::_fw::errors::ExitIf((cond), (msg), __FILE__, __LINE__)
#else //#if CXM_IS_CUDA_AVAILABLE
    #define CXM_DEVICE_ASSERT(cond, msg)
#endif //#if CXM_IS_CUDA_AVAILABLE

#endif //CORTEXMIND_FRAMEWORK_TOOLS_ERRORS_HPP