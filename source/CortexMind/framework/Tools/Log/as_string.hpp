//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_LOG_AS_STRING_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_LOG_AS_STRING_HPP

#include <CortexMind/framework/Tools/Log/level.hpp>
#include <tlx/string.hpp>

namespace cortex::_fw {
    /**
     * @brief Returns the textual representation of a log level.
     *
     * @param level Log severity level.
     *
     * @return Canonical string representation of the specified level.
     */
    [[nodiscard]]
    tlx::vstring as_string(LogLevel level);
    /**
     * @brief Returns the ANSI escape sequence associated with a log level.
     *
     * The returned escape sequence may be used to color terminal output.
     *
     * @param level Log severity level.
     *
     * @return ANSI escape sequence corresponding to the specified level.
     */
    [[nodiscard]]
    tlx::vstring ansi(LogLevel level);
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_LOG_AS_STRING_HPP