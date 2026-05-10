//
// Created by muham on 10.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_ARENA_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_ARENA_HPP

namespace cortex::_fw::sys {
    /**
     * @brief Represents a single memory block in an arena allocator.
     *
     * Used by arena-based memory managers to track allocated regions,
     * their sizes, and usage status.
     */
    struct Arena {
        size_t offset;   ///< Offset from the base pointer of the arena
        size_t size;     ///< Size of this memory block in bytes
        bool   used;     ///< Whether this block is currently in use
    };
} //namespace cortex::_fw::sys

#endif //CORTEXMIND_FRAMEWORK_MEMORY_ARENA_HPP