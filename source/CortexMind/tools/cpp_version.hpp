//
// Created by muham on 14.04.2026.
//

#ifndef CORTEXMIND_TOOLS_CPP_VERSION_HPP
#define CORTEXMIND_TOOLS_CPP_VERSION_HPP

#include <CortexMind/tools/params.hpp>

namespace cortex {
    /**
     * @brief Writes C++ version you use on console
     */
     u0 cpp_version();
} //namespace cortex

/**
 * @brief Writes C++ version you use on console
 */
#define CXM_CPP_VERSION() \
    (::cortex::cpp_version())

#endif //CORTEXMIND_TOOLS_CPP_VERSION_HPP