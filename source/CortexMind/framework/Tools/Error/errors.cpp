//
// Created by muham on 3.08.2026.
//

#include "errors.hpp"
#include <CortexMind/framework/Tools/Log/as_string.hpp>
#include <CortexMind/framework/Tools/Log/w.hpp>
#include <CortexMind/framework/Tools/console.hpp>
#include <CortexMind/runtime/macros.hpp>
#include <iostream>

using namespace cortex::_fw;

void errors::ExitIf(const bool condition, const tlx::vstring msg, const std::source_location &sloc) {
    if (condition) {
        CXM_INITIALIZE_CONSOLE();
        constexpr auto level = LogLevel::ERROR;
        WLog(level)
        << "[" << as_string(level) << "] " << "\n"
        << "[" << sloc.file_name() << ":" << sloc.line() << "] | " << msg << std::endl;
        std::exit(CXM_ERR_EXIT);
    }
}

void errors::WarnIf(const bool condition, const tlx::vstring msg, const std::source_location &sloc) {
    if (condition) {
        CXM_INITIALIZE_CONSOLE();
        constexpr auto level = LogLevel::WARNING;
        WLog(level)
        << "[" << as_string(level) << "] " << "\n"
        << "[" << sloc.file_name() << ":" << sloc.line() << "] | " << msg << std::endl;
    }
}

void errors::WrongDevice(const std::source_location& sloc) {
    CXM_INITIALIZE_CONSOLE();
    constexpr auto level = LogLevel::WARNING;
    WLog(level)
    << "[" << as_string(level) << "] " << "\n"
    << "[" << sloc.file_name() << ":" << sloc.line() << "] | You're using unknown device" << std::endl;
    std::exit(CXM_ERR_EXIT);
}