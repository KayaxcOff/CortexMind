//
// Created by muham on 17.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_MEM_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_MEM_HPP

#include <CortexMind/framework/Memory/arena.hpp>
#include <CortexMind/framework/Tools/params.hpp>
#include <map>
#include <mutex>

namespace cortex::_fw::sys {
    /**
     * @brief Custom memory allocator with arena-based allocation and coalescing.
     *
     * Manages a single large pre-allocated buffer and tracks allocated/free regions
     * using a map of arenas. Supports aligned allocation and automatic coalescing
     * of adjacent free blocks to reduce fragmentation.
     *
     * Thread-safe via an internal mutex.
     */
    class TrackedMem {
    public:
        explicit TrackedMem(size_t capacity);
        TrackedMem(const TrackedMem&) = delete;
        TrackedMem(TrackedMem&&) = delete;
        ~TrackedMem();

        /**
         * @brief Allocates memory for the requested number of float elements.
         * @param count      Number of f32 elements to allocate
         * @param alignment  Alignment requirement in bytes (default: 32 for AVX2)
         * @return Pointer to the allocated memory, or nullptr if allocation fails
         */
        [[nodiscard]]
        f32* allocate(size_t count, size_t alignment = 32);
        /**
         * @brief Deallocates a previously allocated block.
         * @param ptr Pointer previously returned by `allocate()`
         */
        void deallocate(const f32* ptr);
        /**
         * @brief Resets the allocator, freeing all allocations and returning to initial state.
         */
        void reset();
        /**
         * @brief Returns the total capacity of the allocator (in number of f32 elements).
         */
        [[nodiscard]]
        size_t capacity() const noexcept;
        /**
         * @brief Returns the number of currently allocated f32 elements.
         */
        [[nodiscard]]
        size_t used() const noexcept;
    private:
        f32* m_buffer;
        size_t m_capacity;
        size_t m_used;

        std::mutex m_mutex;

        std::map<size_t, Arena> m_arenas;

        /**
         * @brief Merges adjacent free arenas to reduce fragmentation.
         * @param it Iterator pointing to a newly freed arena
         */
        void coalesce(std::map<size_t, Arena>::iterator it);
    };
} //namespace cortex::_fw::sys

#endif //CORTEXMIND_FRAMEWORK_MEMORY_MEM_HPP