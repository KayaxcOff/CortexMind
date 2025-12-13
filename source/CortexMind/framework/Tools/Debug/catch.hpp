//
// Created by muham on 10.12.2025.
//

#ifndef CORTEXMIND_CATCH_HPP
#define CORTEXMIND_CATCH_HPP

#define CXM_ASSERT(isValid, message) \
    do { \
        if ((isValid)) { \
            std::cerr << "[ERROR] " << (message) << std::endl; \
            std::cout << "File: " << __FILE__ << std::endl; \
            std::cout << "Line: " << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

#endif //CORTEXMIND_CATCH_HPP