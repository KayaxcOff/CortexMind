//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_ARENA_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_ARENA_HPP

#include <cwchar>

namespace cortex::_fw {
    struct Arena {
        std::size_t offset;
        std::size_t size;
        bool used;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_MEMORY_ARENA_HPP