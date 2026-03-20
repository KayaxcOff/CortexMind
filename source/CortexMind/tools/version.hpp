//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_TOOLS_VERSION_HPP
#define CORTEXMIND_TOOLS_VERSION_HPP

#include <CortexMind/core/Tools/defaults.hpp>
#include <iostream>

namespace cortex {
    /**
     * @brief   Prints the current CortexMind version to std::cout
     *
     * Format: "CortexMind vMAJOR.MINOR.PATCH"
     *
     * @note    Uses macros CXM_MAJOR, CXM_MINOR, CXM_PATCH
     * @note    No newline at the end — caller can add if needed
     */
    inline void PrintVersion() {
        std::cout << "CortexMind v" << CXM_MAJOR << "." << CXM_MINOR << "." << CXM_PATCH << std::endl;
    }
} // namespace cortex

#endif //CORTEXMIND_TOOLS_VERSION_HPP