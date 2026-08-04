//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_LOG_LEVEL_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_LOG_LEVEL_HPP

#include <cstdint>

namespace cortex::_fw {
    /**
     * @brief Represents the severity level of a log message.
     *
     * LogLevel controls how diagnostic messages are categorized and
     * displayed throughout the framework.
     */
    enum class LogLevel : std::uint8_t {
        ERROR,
        WARNING,
        INFO
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_LOG_LEVEL_HPP