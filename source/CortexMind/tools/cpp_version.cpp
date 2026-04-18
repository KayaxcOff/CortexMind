//
// Created by muham on 14.04.2026.
//

#include "CortexMind/tools/cpp_version.hpp"
#include <iostream>

using namespace cortex;

u0 cortex::cpp_version() {
    std::cout << "C++ version: ";

    #if __cplusplus == 199711L
        std::cout << "C++98/03";
    #elif __cplusplus == 201103L //#if __cplusplus == 199711L
        std::cout << "C++11";
    #elif __cplusplus == 201402L //#elif __cplusplus == 201103L
        std::cout << "C++14";
    #elif __cplusplus == 201703L //#elif __cplusplus == 201402L
        std::cout << "C++17";
    #elif __cplusplus == 202002L //#elif __cplusplus == 201703L
        std::cout << "C++20";
    #elif __cplusplus == 202302L //#elif __cplusplus == 202002L
        std::cout << "C++23";
    #else //#elif __cplusplus == 202302L
        std::cout << "Unknown version (" << __cplusplus << ")";
    #endif //#elif __cplusplus == 202302L #else

    std::cout << std::endl;
}
