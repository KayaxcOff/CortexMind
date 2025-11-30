//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_ASSERT_HPP
#define CORTEXMIND_ASSERT_HPP

#include <iostream>
#include <cstdlib>

#ifdef CORTEX_ENABLE_ASSERTS

    #define CORTEX_ASSERT(condition, msg)                             \
    do {                                                           \
    if (!(condition)) {                                        \
    std::cerr << "[ASSERT FAILED] " << msg                 \
    << "\nFile: " << __FILE__                    \
    << "\nLine: " << __LINE__ << std::endl;      \
    std::abort();                                          \
    }                                                          \
    } while (0)

    #define CORTEX_ASSERT_ZERO(x, y, msg) \
    CORTEX_ASSERT(((x) == 0 || (y) == 0), msg)

    #define CORTEX_ASSERT_EQ(x, y, msg) \
    CORTEX_ASSERT((x) == (y), msg)

#else
    #define CORTEX_ASSERT(condition, msg)      ((void)0)
    #define CORTEX_ASSERT_ZERO(x, y, msg)      ((void)0)
    #define CORTEX_ASSERT_EQ(x, y, msg)        ((void)0)

#endif // CORTEX_ENABLE_ASSERTS

#endif // CORTEXMIND_ASSERT_HPP
