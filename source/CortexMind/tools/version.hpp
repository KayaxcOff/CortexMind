//
// Created by muham on 22.02.2026.
//

#ifndef CORTEXMIND_TOOLS_VERSION_HPP
#define CORTEXMIND_TOOLS_VERSION_HPP

#define CXM_MAJOR 3
#define CXM_MINOR 0
#define CXM_PATCH 0

#include <iostream>

namespace cortex {
    inline
    void PrintVersion() {
        std::cout << "CortexMind v" << CXM_MAJOR << "." << CXM_MINOR << "." << CXM_PATCH << std::endl;
    }
} // namespace cortex

#endif //CORTEXMIND_TOOLS_VERSION_HPP