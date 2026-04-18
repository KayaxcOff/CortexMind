//
// Created by muham on 13.04.2026.
//

#include "CortexMind/tools/version.hpp"
#include <CortexMind/runtime/macros.hpp>
#include <iostream>

cortex::u0 cortex::PrintVersion() {
    std::cout << "CXM-Version: " << CXM_VERSION_MAJOR << "." << CXM_VERSION_MINOR << "." << CXM_VERSION_PATCH << std::endl;
}