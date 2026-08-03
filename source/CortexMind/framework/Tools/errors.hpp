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
    /**
     * @brief Terminates the application if the specified condition evaluates to true.
     *
     * When the condition is satisfied, an error message together with
     * the source file and line number is printed before terminating
     * the process.
     *
     * @param condition Condition indicating whether execution should terminate.
     * @param msg Error message describing the failure.
     * @param sloc Source location of the assertion.
     */
    void ExitIf(bool condition, tlx::vstring msg, const std::source_location& sloc = std::source_location::current());
    /**
     * @brief Emits a runtime warning if the specified condition evaluates to true.
     *
     * Unlike ExitIf(), this function does not terminate the application.
     * The warning is printed together with its source location.
     *
     * @param condition Warning condition.
     * @param msg Warning message.
     * @param sloc Source location of the warning.
     */
    void WarnIf(bool condition, tlx::vstring msg, const std::source_location& sloc = std::source_location::current());
    #if CXM_IS_CUDA_AVAILABLE
        /**
         * @brief Terminates the application if a CUDA runtime call fails.
         *
         * Checks the supplied CUDA error code against cudaSuccess.
         * When an error is detected, a formatted diagnostic message
         * including the CUDA runtime error description, source file,
         * and line number is printed before terminating the application.
         *
         * @param condition CUDA runtime error code.
         * @param msg User-defined error description.
         * @param file Source file where the CUDA call originated.
         * @param line Source line where the CUDA call originated.
         */
        void ExitIf(cudaError_t condition, tlx::vstring msg, tlx::vstring file, int line);
    #endif //#if CXM_IS_CUDA_AVAILABLE

    /**
     * @brief Terminates the application due to an invalid device usage.
     *
     * Reports a fatal runtime error when an operation is attempted on an
     * unsupported or incompatible execution device.
     *
     * This function is intended for internal framework use and is typically
     * invoked through the @c CXM_DEVICE_ERROR macro.
     */
    void WrongDevice();
} //namespace cortex::_fw::errors

/// Runtime assertion for host-side code.
#define CXM_ASSERT(cond, msg) \
    ::cortex::_fw::errors::ExitIf((cond), (msg))

/// Emits a runtime warning without terminating execution.
#define CXM_WARN(cond, msg) \
    ::cortex::_fw::errors::WarnIf((cond), (msg))

#if CXM_IS_CUDA_AVAILABLE
    /// Runtime assertion for CUDA API calls.
    #define CXM_DEVICE_ASSERT(cond, msg) \
        ::cortex::_fw::errors::ExitIf((cond), (msg), __FILE__, __LINE__)
#else //#if CXM_IS_CUDA_AVAILABLE
    #define CXM_DEVICE_ASSERT(cond, msg)
#endif //#if CXM_IS_CUDA_AVAILABLE

/// Reports a fatal device mismatch error.
#define CXM_DEVICE_ERROR() \
    ::cortex::_fw::errors::WrongDevice()

#endif //CORTEXMIND_FRAMEWORK_TOOLS_ERRORS_HPP