//
// Created by muham on 22.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_MEMORY_ARENA_HPP
#define CORTEXMIND_CORE_ENGINE_MEMORY_ARENA_HPP

namespace cortex::_fw::sys {
    struct Arena {
        void* ptr;
        size_t size;
        size_t offset;
    };
} // namespace cortex::_fw::sys

#endif //CORTEXMIND_CORE_ENGINE_MEMORY_ARENA_HPP