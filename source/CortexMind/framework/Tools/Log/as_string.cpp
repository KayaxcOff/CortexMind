//
// Created by muham on 4.08.2026.
//

#include "CortexMind/framework/Tools/Log/as_string.hpp"

namespace {
    constexpr char kRed[]       = "\033[31m";
    constexpr char kGreen[]     = "\033[32m";
    constexpr char kYellow[]    = "\033[33m";
    constexpr char kBlue[]      = "\033[34m";
} //unnamed namespace

tlx::vstring cortex::_fw::as_string(const LogLevel level) {
    switch (level) {
        case LogLevel::ERROR:
            return "ERROR";
        case LogLevel::WARNING:
            return "WARNING";
        case LogLevel::INFO:
            return "INFO";
        default:
            return "DEFAULT";
    }
}

tlx::vstring cortex::_fw::ansi(const LogLevel level) {
    switch (level) {
        case LogLevel::ERROR:
            return kRed;
        case LogLevel::WARNING:
            return kYellow;
        case LogLevel::INFO:
            return kBlue;
        default:
            return kGreen;
    }
}