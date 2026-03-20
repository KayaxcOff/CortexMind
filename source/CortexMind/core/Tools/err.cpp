//
// Created by muham on 13.03.2026.
//

#include "CortexMind/core/Tools/err.hpp"
#include <CortexMind/core/Tools/defaults.hpp>
#include <iostream>

using namespace cortex::_fw;

void err::exitIf(const bool status, const std::string &address, const std::string &message) {
    if (!status) {
        std::cerr << "\033[1;31m" << address << ": " << message << "\033[0m\n";
        std::exit(CXM_ERR_EXIT);
    }
}

void err::warnIf(const bool status, const std::string &address, const std::string &message) {
    if (!status) {
        std::cerr << "\033[1;31m" << address << ": " << message << "\033[0m\n";
    }
}
