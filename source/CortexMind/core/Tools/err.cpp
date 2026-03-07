//
// Created by muham on 21.02.2026.
//

#include "CortexMind/core/Tools/err.hpp"
#include <CortexMind/core/Tools/defaults.hpp>
#include <iostream>

using namespace cortex::_fw;

void err::IsStatusOk(const bool status, const std::string &name, const std::string &msg) {
    if (!status) {
        std::cerr << "Error at " << name << std::endl;
        std::cerr << msg << std::endl;
        std::exit(CXM_ERR_EXIT);
    }
}