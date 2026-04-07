//
// Created by muham on 8.04.2026.
//

#include "CortexMind/framework/Tools/err.hpp"
#include <CortexMind/runtime/macros.hpp>
#include <iostream>

using namespace cortex::_fw;

void err::exitIf(const bool status, const std::string &where, const std::string &message) {
    if (!status) {
        std::cerr << "[CXM-Error]: " << where << std::endl;
        std::cerr << message << std::endl;
        std::exit(CXM_ERR_EXIT);
    }
}

void err::warnIf(const bool status, const std::string &where, const std::string &message) {
    if (!status) {
        std::cerr << "[CXM-Error]: " << where << std::endl;
        std::cerr << message << std::endl;
    }
}