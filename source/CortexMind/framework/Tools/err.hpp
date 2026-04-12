//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_ERR_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_ERR_HPP

#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/core/Engine/CUDA/params.h>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <string>

namespace cortex::_fw {
    /**
     * @brief Error handling and assertion utilities.
     *
     * Provides centralized error checking, CUDA error handling,
     * and warning mechanisms with consistent output formatting.
     */
    struct err {
        /**
         * @brief Exits the program if the status is false.
         * @param status  Condition to check (program exits if `false`)
         * @param where   Location or function name where the error occurred
         * @param message Detailed error message
         */
        static void exitIf(bool status, const std::string& where, const std::string& message);
        #if CXM_IS_CUDA_AVAILABLE
            /**
             * @brief Checks CUDA runtime error and exits if not successful.
             * @param error CUDA error code returned by a CUDA API call
             * @param where Location or function name where the error occurred
             */
            static void exitIf(cudaError_t error, const std::string& where);
        #endif //#if CXM_IS_CUDA_AVAILABLE
        /**
         * @brief Prints a warning message if the status is false.
         * @param status  Condition to check (warning printed if `false`)
         * @param where   Location or function name
         * @param message Warning message
         */
        static void warnIf(bool status, const std::string& where, const std::string& message);
    };
} //namespace cortex::_fw

/**
 * @brief Macro for fatal assertion with custom message.
 */
#define CXM_ASSERT(status, whr, msg) \
    (::cortex::_fw::err::exitIf((status), (whr), (msg)))

/**
 * @brief Macro for non-fatal warning.
 */
#define CXM_WARN(status, whr, msg) \
    (::cortex::_fw::err::warnIf((status), (whr), (msg)))

/**
 * @brief Macro for CUDA error checking.
 */
#if CXM_IS_CUDA_AVAILABLE
    #define CXM_CUDA_ASSERT(status, whr) \
        (::cortex::_fw::err::exitIf((status), (whr)))
#else //#if CXM_IS_CUDA_AVAILABLE
    #define CXM_CUDA_ASSERT(status, whr) ((void)0)
#endif //#if CXM_IS_CUDA_AVAILABLE #else

#endif //CORTEXMIND_FRAMEWORK_TOOLS_ERR_HPP