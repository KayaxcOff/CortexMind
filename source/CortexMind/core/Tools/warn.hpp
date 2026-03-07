//
// Created by muham on 22.02.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_WARN_HPP
#define CORTEXMIND_CORE_TOOLS_WARN_HPP

#include <iostream>

#define CXM_WARN(name, msg)                                         \
    do {                                                            \
        std::cerr << "[CXM Warning] " << (name) << "\n"     \
        << "  " << (msg) << "\n";                         \
    } while(0)


#define CXM_WARN_IF(condition, name, msg)                           \
    do {                                                            \
        if (!(condition)) CXM_WARN((name), (msg));                  \
    } while(0)

#endif //CORTEXMIND_CORE_TOOLS_WARN_HPP