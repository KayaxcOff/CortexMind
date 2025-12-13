//
// Created by muham on 10.12.2025.
//

#ifndef CORTEXMIND_LOG_HPP
#define CORTEXMIND_LOG_HPP

#include <iostream>

namespace cortex {
    inline void log(const std::string &message) {
        std::cout << "[LOG]: " << message << std::endl;
    }
}

#endif //CORTEXMIND_LOG_HPP