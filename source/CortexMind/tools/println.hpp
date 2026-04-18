//
// Created by muham on 13.04.2026.
//

#ifndef CORTEXMIND_TOOLS_PRINTLN_HPP
#define CORTEXMIND_TOOLS_PRINTLN_HPP

#include <CortexMind/tools/params.hpp>
#include <iostream>
#include <format>

namespace cortex {
    /**
     * @brief Prints formatted output without a trailing newline.
     *
     * Uses `std::format` to format the arguments according to the format string.
     *
     * @tparam Args Variadic template parameter pack for format arguments
     * @param fmt   Format string (std::format_string)
     * @param args  Arguments to be formatted and printed
     *
     * @code
     * print("Epoch {} - Loss: {:.4f}", epoch, loss);
     * @endcode
     */
    template<typename... Args>
    u0 print(std::format_string<Args...> fmt, Args&&... args) {
        std::cout << std::format(fmt, std::forward<Args>(args)...);
    }

    /**
     * @brief Prints formatted output and appends a newline (`\n`).
     *
     * This is a convenience wrapper that calls `print()` and then outputs a newline.
     *
     * @tparam Args Variadic template parameter pack for format arguments
     * @param fmt   Format string
     * @param args  Arguments to be formatted and printed
     *
     * @code
     * println("Accuracy: {:.2f}%", accuracy * 100);
     * println("Training completed in {} seconds", elapsed);
     * @endcode
     */
    template<typename... Args>
    u0 println(std::format_string<Args...> fmt, Args&&... args) {
        print(fmt, std::forward<Args>(args)...);
        std::cout << '\n';
    }
} //namespace cortex

#endif //CORTEXMIND_TOOLS_PRINTLN_HPP