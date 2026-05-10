//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_ERR_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_ERR_HPP

#include <CortexMind/framework/Tools/types.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <cuda_runtime.h>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <string_view>

namespace cortex::_fw {
    /**
     * @brief Error handling and assertion utilities.
     *
     * Provides runtime assertion, warning and CUDA error checking facilities.
     * All functions are designed to be lightweight with minimal overhead
     * when the condition is false.
     */
    struct err {
        /**
         * @brief Checks condition and terminates program if true.
         *
         * @param condition If `true`, program will print error message and abort.
         * @param message   Error message to display.
         * @param file      Source file name (usually `__FILE__`).
         * @param line      Line number (usually `__LINE__`).
         *
         * @note Uses `std::abort()` after printing to stderr.
         */
        static void exitIf(bool condition, const std::string_view& message, const char* file, i32 line);
        /**
         * @brief Issues a warning if condition is true.
         *
         * @param condition If `true`, prints warning message to stderr.
         * @param message   Warning message.
         * @param file      Source file name.
         * @param line      Line number.
         *
         * @note Program execution continues after warning.
         */
        static void warnIf(bool condition, const std::string_view& message, const char* file, i32 line);
        #if CXM_IS_CUDA_AVAILABLE
            /**
             * @brief Checks CUDA runtime error and terminates if not successful.
             *
             * @param call    CUDA API return value (`cudaError_t`).
             * @param message Descriptive error message.
             * @param file    Source file name.
             * @param line    Line number.
             *
             * @note Only available when `CXM_IS_CUDA_AVAILABLE` is defined.
             */
            static void exitIf_c(cudaError_t call, const char* file, i32 line);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    };
} //namespace cortex::_fw

/**
 * @def CXM_ASSERT(cond, msg)
 * @brief Runtime assertion macro. Terminates program if condition is true.
 */
#define CXM_ASSERT(cond, msg) \
    ::cortex::_fw::err::exitIf((cond), (msg), __FILE__, __LINE__)

/**
 * @def CXM_WARN(cond, msg)
 * @brief Warning macro. Prints warning if condition is true.
 */
#define CXM_WARN(cond, msg) \
    ::cortex::_fw::err::warnIf((cond), (msg), __FILE__, __LINE__)

#if CXM_IS_CUDA_AVAILABLE
    /**
     * @def CXM_CUDA_ASSERT(call, msg)
     * @brief CUDA error checking macro.
     */
    #define CXM_CUDA_ASSERT(call) \
        ::cortex::_fw::err::exitIf_c((call), __FILE__, __LINE__)
#else //#if CXM_IS_CUDA_AVAILABLE
    #define CXM_CUDA_ASSERT(call, msg) ((void)(0))
#endif //#if CXM_IS_CUDA_AVAILABLE #else

#endif //CORTEXMIND_FRAMEWORK_TOOLS_ERR_HPP