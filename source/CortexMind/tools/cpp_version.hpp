//
// Created by muham on 14.04.2026.
//

#ifndef CORTEXMIND_TOOLS_CPP_VERSION_HPP
#define CORTEXMIND_TOOLS_CPP_VERSION_HPP

namespace cortex {
    /**
     * @brief Writes C++ version you use on console
     */
    void cpp_version();
} //namespace cortex

/**
 * @brief Writes C++ version you use on console
 */
#define CXM_CPP_VERSION() \
    (::cortex::cpp_version())

#endif //CORTEXMIND_TOOLS_CPP_VERSION_HPP