//
// Created by muham on 16.03.2026.
//

#ifndef CORTEXMIND_TOOLS_PRINTLN_HPP
#define CORTEXMIND_TOOLS_PRINTLN_HPP

#include <concepts>
#include <iostream>
#include <type_traits>

namespace cortex {
    /**
     * @concept StreamInsertable
     * @brief   Concept that checks if a type can be inserted into std::ostream
     */
    template<typename T>
    concept StreamInsertable = requires(std::ostream& os, T value) {
        { os << value } -> std::same_as<std::ostream&>;
    };

    /**
     * @brief   Prints a single value to std::cout (no newline)
     * @tparam  T       Type that satisfies StreamInsertable concept
     * @param   value   Value to print
     */
    template<StreamInsertable T>
    void print(T&& value) {
        std::cout << std::forward<T>(value);
    }

    /**
     * @brief   Recursively prints multiple values with space separation (no final newline)
     * @tparam  T       First value type
     * @tparam  Args    Remaining value types
     * @param   value   First value
     * @param   args    Remaining values
     */
    template<StreamInsertable T, StreamInsertable... Args>
    void print(T&& value, Args&&... args) {
        std::cout << std::forward<T>(value) << " ";
        print(std::forward<Args>(args)...);
    }

    /**
     * @brief   Prints multiple values with space separation and adds a newline
     * @tparam  Args    Value types (must all satisfy StreamInsertable)
     * @param   args    Values to print
     *
     * @note    Equivalent to print(...) followed by std::cout << '\n';
     */
    template<StreamInsertable... Args>
    void println(Args&&... args) {
        print(std::forward<Args>(args)...);
        std::cout << '\n';
    }
} // namespace cortex

#endif //CORTEXMIND_TOOLS_PRINTLN_HPP