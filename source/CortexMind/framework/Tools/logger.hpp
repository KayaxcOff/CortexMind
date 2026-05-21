//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_LOGGER_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_LOGGER_HPP

#include <iomanip>
#include <string>

namespace cortex::_fw {
    /**
     * @brief Log severity levels.
     */
    enum class LogLevel {
        TRACE,   ///< Most verbose level - for detailed tracing
        DEBUG,   ///< Debug information
        WARN,    ///< Warning messages
        ERROR    ///< Error messages
    };

    /**
     * @brief Singleton logger class for the framework.
     *
     * Provides simple logging functionality with configurable minimum severity
     * level and enable/disable control.
     */
    class Logger {
    public:
        /**
         * @brief Returns the singleton instance of the logger.
         */
        static Logger& getInstance();

        /**
         * @brief Sets the minimum log level.
         *
         * Messages below this level will be ignored.
         */
        void setLevel(LogLevel level);
        /**
         * @brief Enables or disables all logging output.
         */
        void setEnabled(bool e);

        /** @brief Logs a message at TRACE level. */
        void trace(const std::string& msg) const;

        /** @brief Logs a message at DEBUG level. */
        void debug(const std::string& msg) const;

        /** @brief Logs a message at WARN level. */
        void warn(const std::string& msg) const;

        /** @brief Logs a message at ERROR level. */
        void error(const std::string& msg) const;

    private:
        LogLevel min_level;
        bool enabled;

        Logger();
        /**
         * @brief Internal logging function.
         *
         * Checks level and enabled state, then prints the message with prefix.
         */
        void log(LogLevel level, const std::string& msg) const;
    };

    /**
     * @brief Checks if a floating point value is NaN (Not a Number).
     */
    inline bool isNaN(const float val) {
        return std::isnan(val);
    }
    /**
     * @brief Checks if a floating point value is infinite (±∞).
     */
    inline bool isInf(const float val) {
        return std::isinf(val);
    }
    /**
     * @brief Checks if a floating point value is anomalous.
     *
     * Returns true if the value is NaN, Inf, or has extremely large magnitude.
     */
    inline bool isAnomalous(const float val) {
        return isNaN(val) || isInf(val) || std::abs(val) > 1e10f;
    }
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_LOGGER_HPP