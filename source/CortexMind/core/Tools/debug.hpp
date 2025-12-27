#ifndef CORTEXMIND_CORE_TOOLS_DEBUG_HPP
#define CORTEXMIND_CORE_TOOLS_DEBUG_HPP

#include <iostream>
#include <string>
#include <cstdlib>

namespace cortex::_fw::err {
    inline void IsStatusFailed(bool isFailed, const std::string& fileName, const std::string& errorMessage) {
        if (isFailed) {
            std::cerr << "Error at " << fileName << "\n" << errorMessage << std::endl;
            std::exit(1);
        }
    }
} // namespace cortex::_fw::err

#endif // CORTEXMIND_CORE_TOOLS_DEBUG_HPP