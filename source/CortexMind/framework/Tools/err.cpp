//
// Created by muham on 31.03.2026.
//

#include "CortexMind/framework/Tools/err.hpp"
#include <CortexMind/runtime/Flags/defaults.hpp>
#include <iostream>

using namespace cortex::_fw;

void err::exitIf(const bool status, const std::string &message, const std::string &where) {
    if (!status) {
        std::cerr << "Error at " << where << std::endl;
        std::cerr << message << std::endl;
        std::exit(CXM_ERR_EXIT);
    }
}

void err::warnIf(const bool status, const std::string &message, const std::string &where) {
    if (!status) {
        std::cerr << "Error at " << where << std::endl;
        std::cerr << message << std::endl;
    }
}