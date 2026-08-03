//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_MEM_HPP
#define CORTEXMIND_MEM_HPP

#include <CortexMind/framework/Memory/arena.hpp>
#include <cstddef>
#include <map>
#include <mutex>

namespace cortex::_fw {
    /**
     * @brief Arena-based memory allocator with allocation tracking.
     *
     * TrackedMem manages a contiguous block of host memory and provides
     * aligned allocation and deallocation without relying on the system
     * allocator for every request.
     *
     * Memory is internally divided into blocks represented by @ref Arena.
     * Free blocks are automatically split during allocation and merged
     * during deallocation to reduce fragmentation.
     *
     * The allocator is thread-safe and serializes allocation requests
     * using an internal mutex.
     *
     * Typical use cases include tensor storage, temporary workspaces,
     * and other high-frequency runtime allocations.
     */
    class TrackedMem {
    public:
        /**
         * @brief Creates a tracked memory arena.
         *
         * Allocates a contiguous host memory buffer with the specified capacity.
         *
         * @param bytes Total capacity of the arena in bytes.
         */
        explicit TrackedMem(std::size_t bytes);
        TrackedMem(const TrackedMem &) = delete;
        TrackedMem(TrackedMem &&) = delete;
        ~TrackedMem();

        /**
         * @brief Allocates an aligned memory block.
         *
         * Searches the arena for a suitable free block, applies the requested
         * alignment, and splits the block if necessary.
         *
         * @param count Number of bytes to allocate.
         * @param alignment Required memory alignment.
         *
         * @return Pointer to the allocated memory, or nullptr if no suitable
         * block exists.
         */
        [[nodiscard]]
        std::byte* allocate(std::size_t count, std::size_t alignment = 32);
        /**
         * @brief Releases a previously allocated memory block.
         *
         * Marks the block as free and attempts to merge adjacent free blocks
         * to reduce memory fragmentation.
         *
         * @param ptr Pointer previously returned by allocate().
         */
        void deallocate(const std::byte* ptr);
        /**
         * @brief Resets the allocator to its initial state.
         *
         * All allocations become invalid and the entire arena is marked as free.
         */
        void reset();
        /**
         * @brief Returns the total capacity of the arena.
         */
        [[nodiscard]]
        std::size_t capacity() const noexcept;
        /**
         * @brief Returns the number of bytes currently allocated.
         */
        [[nodiscard]]
        std::size_t used() const noexcept;
    private:
        std::byte* m_buffer;
        std::size_t m_capacity;
        std::size_t m_used;

        std::mutex m_mutex;

        std::map<std::size_t, Arena> m_arenas;

        /**
         * @brief Merges adjacent free blocks.
         *
         * Attempts to merge the specified block with its neighboring
         * free blocks to reduce fragmentation.
         *
         * @param it Iterator referencing the block to merge.
         */
        void coalesce(std::map<std::size_t, Arena>::iterator it);
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_MEM_HPP