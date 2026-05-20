//
// Created by muham on 20.05.2026.
//

#ifndef CORTEXMIND_TOOLS_PRINTLN_HPP
#define CORTEXMIND_TOOLS_PRINTLN_HPP

#include <iostream>
#include <format>
#include <type_traits>

namespace cortex {
    template <typename... Args>
    void println(std::format_string<Args...> fmt, Args&&... args) {
        std::cout << std::format(fmt, std::forward<Args>(args)...) << std::endl;
    }
} //namespace cortex

#endif //CORTEXMIND_TOOLS_PRINTLN_HPP