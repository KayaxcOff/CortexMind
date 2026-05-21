//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_LOGGER_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_LOGGER_HPP

#include <iostream>
#include <string>
#include <iomanip>

namespace cortex::_fw {
    enum class LogLevel {
        TRACE,
        DEBUG,
        WARN,
        ERROR
    };

    class Logger {
        LogLevel min_level = LogLevel::DEBUG;
        bool enabled = true;

        Logger() = default;

    public:
        static Logger& getInstance() {
            static Logger instance_obj;  // ← Local static, garantili single initialization
            return instance_obj;
        }

        void setLevel(const LogLevel level) { min_level = level; }
        void setEnabled(const bool e) { enabled = e; }

        void trace(const std::string& msg) const { log(LogLevel::TRACE, msg); }
        void debug(const std::string& msg) const { log(LogLevel::DEBUG, msg); }
        void warn(const std::string& msg) const { log(LogLevel::WARN, msg); }
        void error(const std::string& msg) const { log(LogLevel::ERROR, msg); }

    private:
        void log(const LogLevel level, const std::string& msg) const {
            if (!enabled || level < min_level) return;

            std::string prefix;
            switch (level) {
                case LogLevel::TRACE: prefix = "[TRACE] "; break;
                case LogLevel::DEBUG: prefix = "[DEBUG] "; break;
                case LogLevel::WARN:  prefix = "[WARN]  "; break;
                case LogLevel::ERROR: prefix = "[ERROR] "; break;
            }
            std::cout << prefix << msg << std::endl;
        }
    };

    inline bool isNaN(const float val) { return std::isnan(val); }
    inline bool isInf(const float val) { return std::isinf(val); }
    inline bool isAnomalous(const float val) {
        return isNaN(val) || isInf(val) || std::abs(val) > 1e10f;
    }
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_LOGGER_HPP