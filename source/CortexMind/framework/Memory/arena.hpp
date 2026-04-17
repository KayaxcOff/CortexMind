//
// Created by muham on 17.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_ARENA_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_ARENA_HPP

namespace cortex::_fw::sys {
    /**
     * @brief Represents a single memory block (arena) in the custom allocator.
     */
    struct Arena {
        size_t offset;  ///< Starting offset from the base buffer (in number of f32 elements)
        size_t size;    ///< Size of this arena in number of f32 elements
        bool used;      ///< True if this arena is currently allocated/used
    };
} //namespace cortex::_fw::sys

#endif //CORTEXMIND_FRAMEWORK_MEMORY_ARENA_HPP