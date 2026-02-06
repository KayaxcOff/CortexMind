//
// Created by muham on 4.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_MEMORY_UTIL_HPP
#define CORTEXMIND_CORE_ENGINE_MEMORY_UTIL_HPP

namespace cortex::_fw::sys {
    struct Arena {
        void* ptr;
        size_t size;
        size_t offset;
    };
} // namespace cortex::_fw::sys

#endif //CORTEXMIND_CORE_ENGINE_MEMORY_UTIL_HPP