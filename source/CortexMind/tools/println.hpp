//
// Created by muham on 6.02.2026.
//

#ifndef CORTEXMIND_TOOLS_PRINTLN_HPP
#define CORTEXMIND_TOOLS_PRINTLN_HPP

#include <iostream>
#include <concepts>

namespace cortex {

    /**
     * @concept StreamInsertable
     * @brief Ensures that a type can be inserted into an output stream.
     *
     * This concept constrains types that support the `operator<<`
     * with `std::ostream`.
     */
    template<typename T>
    concept StreamInsertable =
        requires(std::ostream& os, T value) {
            { os << value } -> std::same_as<std::ostream&>;
        };

    /**
     * @brief Prints a single value to the standard output stream.
     *
     * Outputs the given value to `std::cout` without appending
     * a newline character.
     *
     * @tparam T Type of the value to be printed.
     * @param value The value to print.
     *
     * @example
     * @code
     * cortex::print("Hello");
     * cortex::print(42);
     * @endcode
     */
    template<StreamInsertable T>
    void print(T&& value) {
        std::cout << std::forward<T>(value);
    }

    /**
     * @brief Prints multiple values separated by spaces.
     *
     * Recursively prints each argument separated by a space.
     * Does not append a newline character.
     *
     * @tparam T Type of the first value.
     * @tparam Args Types of the remaining values.
     * @param value First value to print.
     * @param args Remaining values to print.
     */
    template<StreamInsertable T, StreamInsertable... Args>
    void print(T&& value, Args&&... args) {
        std::cout << std::forward<T>(value) << " ";
        print(std::forward<Args>(args)...);
    }

    /**
     * @brief Prints values followed by a newline.
     *
     * Behaves like `print(...)` but appends a newline character
     * at the end of the output.
     *
     * @tparam Args Types of the values to be printed.
     * @param args Values to print.
     *
     * @example
     * @code
     * cortex::println("Player score:", 128, "points");
     * cortex::println(3.14);
     * @endcode
     */
    template<StreamInsertable... Args>
    void println(Args&&... args) {
        print(std::forward<Args>(args)...);
        std::cout << '\n';
    }

} // namespace cortex

#endif // CORTEXMIND_TOOLS_PRINTLN_HPP
