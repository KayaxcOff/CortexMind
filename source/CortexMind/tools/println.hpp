//
// Created by muham on 22.02.2026.
//

#ifndef CORTEXMIND_TOOLS_PRINTLN_HPP
#define CORTEXMIND_TOOLS_PRINTLN_HPP

#include <iostream>
#include <concepts>

namespace cortex {
    template<typename T>
    concept StreamInsertable = requires(std::ostream& os, T value) {
        { os << value } -> std::same_as<std::ostream&>;
    };

    template<StreamInsertable T>
    void print(T&& value) {
        std::cout << std::forward<T>(value);
    }

    template<StreamInsertable T, StreamInsertable... Args>
    void print(T&& value, Args&&... args) {
        std::cout << std::forward<T>(value) << " ";
        print(std::forward<Args>(args)...);
    }

    template<StreamInsertable... Args>
    void println(Args&&... args) {
        print(std::forward<Args>(args)...);
        std::cout << '\n';
    }
} // namespace cortex

#endif //CORTEXMIND_TOOLS_PRINTLN_HPP