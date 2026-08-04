//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_LOG_W_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_LOG_W_HPP

#include <CortexMind/framework/Tools/Log/level.hpp>
#include <iostream>
#include <ostream>

namespace cortex::_fw {
    /**
     * @brief Colored output stream wrapper for logging.
     *
     * WLog provides a lightweight RAII-based interface for writing
     * colored log messages to an output stream.
     *
     * The selected ANSI color is applied during construction and
     * automatically reset when the object is destroyed.
     *
     * Typical usage:
     * @code
     * WLog(LogLevel::INFO)
     *     << "Training started";
     * @endcode
     */
    class WLog {
    public:
        explicit WLog(LogLevel level, std::ostream& os = std::cout);
        WLog(const WLog&) = delete;
        WLog(WLog&&) = delete;
        ~WLog() noexcept;

        template<typename T>
        WLog& operator<<(const T& value) {
            this->m_os << value;
            return *this;
        }
        WLog& operator<<(std::ostream& (*manip)(std::ostream&));

        WLog& operator=(const WLog&) = delete;
        WLog& operator=(WLog&&) = delete;
    private:
        std::ostream& m_os;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_LOG_W_HPP