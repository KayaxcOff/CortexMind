//
// Created by muham on 26.02.2026.
//

#ifndef CORTEXMIND_TOOLS_CPP_VERSION_HPP
#define CORTEXMIND_TOOLS_CPP_VERSION_HPP

#include <iostream>

namespace cortex {
    inline
    void checkCompileVersion() {
        std::cout << "C++ Version: " << __cplusplus << std::endl;
    #if __cplusplus >= 202002L
            std::cout << "C++20 or higher is being used..\n";
    #else
            std::cout << "C++20 is not being used..\n";
    #endif
    }
} // namespace cortex

#define CXM_CPP_VERSION() do { ::cortex::checkCompileVersion(); } while(0)

#endif //CORTEXMIND_TOOLS_CPP_VERSION_HPP