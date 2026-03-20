//
// Created by muham on 14.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_MEMORY_UTILS_HPP
#define CORTEXMIND_CORE_ENGINE_MEMORY_UTILS_HPP

namespace cortex::_fw::sys {
    /**
     * @struct  Arena
     * @brief   Represents one large contiguous chunk allocated from cudaMemPool
     */
    struct Arena {
        void* ptr;      ///< Device pointer to start of chunk
        size_t size;    ///< Total size in bytes
        size_t offset;  ///< Current bump allocation offset
    };
    /**
     * @struct  Block
     * @brief   Represents one free block in the free list
     */
    struct Block {
        void* ptr;      ///< Device pointer to start of free block
        size_t size;    ///< Size in bytes (already aligned)
    };

    /**
     * @brief   Rounds up value to the next multiple of alignment
     * @param   value       Value to align
     * @param   alignment   Must be power of 2
     * @return  Smallest value >= input that is multiple of alignment
     *
     * @note    Used for 256-byte alignment (good for Tensor Core / coalesced access)
     */
    static size_t align_up(const size_t value, const size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }
} // namespace cortex::_fw::sys

#endif //CORTEXMIND_CORE_ENGINE_MEMORY_UTILS_HPP