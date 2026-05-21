//
// Created by muham on 21.05.2026.
//

#include "CortexMind/framework/Tools/logger.hpp"
#include <iostream>

using namespace cortex::_fw;

Logger &Logger::getInstance() {
    static Logger instance;
    return instance;
}

void Logger::setLevel(const LogLevel level) {
    this->min_level = level;
}

void Logger::setEnabled(const bool e) {
    this->enabled = e;
}

void Logger::trace(const std::string &msg) const {
    this->log(LogLevel::TRACE, msg);
}

void Logger::debug(const std::string &msg) const {
    this->log(LogLevel::DEBUG, msg);
}

void Logger::warn(const std::string &msg) const {
    this->log(LogLevel::WARN, msg);
}

void Logger::error(const std::string &msg) const {
    this->log(LogLevel::ERROR, msg);
}

Logger::Logger() : min_level(LogLevel::DEBUG), enabled(true) {}

void Logger::log(const LogLevel level, const std::string &msg) const {
    if (!this->enabled || level < this->min_level) {
        return;
    }

    std::string prefix;
    switch (level) {
        case LogLevel::TRACE: prefix = "[TRACE] "; break;
        case LogLevel::DEBUG: prefix = "[DEBUG] "; break;
        case LogLevel::WARN:  prefix = "[WARN]  "; break;
        case LogLevel::ERROR: prefix = "[ERROR] "; break;
    }
    std::cout << prefix << msg << std::endl;
}