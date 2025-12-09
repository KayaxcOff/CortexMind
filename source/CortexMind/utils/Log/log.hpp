//
// Created by muham on 6.12.2025.
//

#ifndef CORTEXMIND_LOG_HPP
#define CORTEXMIND_LOG_HPP

#include <iostream>
namespace cortex {
    inline void log(const std::string &message) {
        std::cout << message << std::endl;
    }
}

#endif //CORTEXMIND_LOG_HPP