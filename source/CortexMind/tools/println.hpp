//
// Created by muham on 20.05.2026.
//

#ifndef CORTEXMIND_TOOLS_PRINTLN_HPP
#define CORTEXMIND_TOOLS_PRINTLN_HPP

#include <iostream>
#include <format>
#include <type_traits>

namespace cortex {
    /**
     * @brief Prints a formatted string to stdout followed by a newline.
     *
     * This is a convenient wrapper around `std::format` and `std::cout`
     * that automatically appends `std::endl`.
     *
     * @tparam Args Types of the format arguments
     * @param fmt   Format string (using std::format syntax)
     * @param args  Arguments to be formatted
     *
     * @example
     * @code
     * println("Hello, {}! The answer is {}.", "world", 42);
     * // Output: Hello, world! The answer is 42.
     * @endcode
     */
    template <typename... Args>
    void println(std::format_string<Args...> fmt, Args&&... args) {
        std::cout << std::format(fmt, std::forward<Args>(args)...) << std::endl;
    }
} //namespace cortex

#endif //CORTEXMIND_TOOLS_PRINTLN_HPP