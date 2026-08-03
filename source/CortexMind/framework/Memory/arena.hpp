//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_ARENA_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_ARENA_HPP

#include <cwchar>

namespace cortex::_fw {
    /**
     * @brief Describes a memory block managed by an arena allocator.
     *
     * Arena stores the metadata associated with a contiguous memory region
     * within an arena-based allocator. Each block records its position,
     * size, and allocation state.
     *
     * The structure is used internally by the framework to manage dynamic
     * memory allocation and deallocation without relying on the system
     * allocator for every request.
     */
    struct Arena {
        std::size_t offset;
        std::size_t size;
        bool used;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_MEMORY_ARENA_HPP