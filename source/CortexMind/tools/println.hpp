//
// Created by muham on 13.04.2026.
//

#ifndef CORTEXMIND_TOOLS_PRINTLN_HPP
#define CORTEXMIND_TOOLS_PRINTLN_HPP

#include <iostream>
#include <format>

namespace cortex {
    /**
     * @brief Prints multiple arguments separated by space and appends a newline.
     *
     * Uses the left fold expression to stream all arguments to `std::cout`.
     *
     * @tparam Args Variadic template parameter pack
     * @param args  Values to print
     *
     * @code
     * println("Hello", "world", 42, 3.14);
     * @endcode
     */
    template<typename... Args>
    void println(Args&&... args) {
        (std::cout << ... << args) << '\n';
    }

    /**
     * @brief Prints a formatted string using `std::format` and appends a newline.
     *
     * This overload supports C++20 `std::format` syntax for type-safe formatting.
     *
     * @tparam Args Variadic template parameter pack for format arguments
     * @param fmt   Format string (std::format_string)
     * @param args  Arguments to be formatted
     *
     * @code
     * println("Epoch: {}, Loss: {:.4f}, Accuracy: {:.2f}%", epoch, loss, acc * 100);
     * @endcode
     */
    template<typename... Args>
    void println(std::format_string<Args...> fmt, Args&&... args) {
        std::cout << std::format(fmt, std::forward<Args>(args)...) << '\n';
    }
} //namespace cortex

#endif //CORTEXMIND_TOOLS_PRINTLN_HPP