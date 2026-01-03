//
// Created by muham on 29.12.2025.
//

#ifndef CORTEXMIND_CORE_TOOLS_DEBUG_HPP
#define CORTEXMIND_CORE_TOOLS_DEBUG_HPP

#include <iostream>
#include <string>

namespace cortex::_fw::err {
    inline void IsAnError(const bool isValid, const std::string& name, const std::string& msg) {
        if (!isValid) {
            std::cerr << name << std::endl;
            std::cerr << msg << std::endl;
            std::exit(1);
        }
    }
} // namespace cortex::_fw::err

#endif //CORTEXMIND_CORE_TOOLS_DEBUG_HPP