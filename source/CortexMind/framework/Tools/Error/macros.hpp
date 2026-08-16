//
// Created by muham on 16.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_ERRORS_MACROS_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_ERRORS_MACROS_HPP

#include <CortexMind/framework/Tools/Log/w.hpp>

using namespace cortex::_fw;

#define CXM_RETURN_IF(cond, msg)                \
    do {                                        \
        if (cond) {                             \
            WLog(LogLevel::INFO) << msg         \
            << "\n";                            \
            return;                             \
        }                                       \
    } while (false)

#define CXM_RETURN_NULLPTR(cond, msg)               \
    do {                                            \
        if (cond) {                                 \
            WLog(LogLevel::INFO) << msg             \
            << "\n";                                \
            return nullptr;                         \
        }                                           \
    } while (false)

#endif //CORTEXMIND_FRAMEWORK_TOOLS_ERRORS_MACROS_HPP