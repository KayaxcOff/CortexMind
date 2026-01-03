//
// Created by muham on 29.12.2025.
//

#ifndef CORTEXMIND_TOOLS_LOG_HPP
#define CORTEXMIND_TOOLS_LOG_HPP

#include <iostream>

namespace cortex {
    template<typename T>
    // @brief Logs a single value to the standard output.
    void log(T&& value) {
        std::cout << value << std::endl;
    }

    template<typename T, typename... Args>
    // @brief Logs multiple values to the standard output.
    void log(T&& first, Args&&... args) {
        std::cout << first << " ";
        log(std::forward<Args>(args)...);
    }
} // namespace cortex

#endif // CORTEXMIND_TOOLS_LOG_HPP