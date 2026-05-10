//
// Created by muham on 10.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_MEM_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_MEM_HPP

#include <CortexMind/framework/Memory/arena.hpp>
#include <CortexMind/framework/Tools/types.hpp>
#include <CortexMind/runtime/macros.hpp>
#include <map>
#include <mutex>

namespace cortex::_fw::sys {
    /**
     * @brief Arena-based memory allocator with tracking and coalescing.
     *
     * This class implements a custom memory allocator optimized for
     * machine learning workloads. It uses a single large pre-allocated buffer
     * and manages allocations using an arena strategy with automatic
     * coalescing of freed adjacent blocks.
     *
     * Features:
     * - Alignment-aware allocation
     * - Memory block tracking
     * - Automatic coalescing on deallocation
     * - Thread-safe operations
     */
    class TrackedMem {
    public:
        /**
         * @brief Constructs a new memory arena.
         *
         * @param capacity Total capacity of the arena in number of `f32` elements.
         */
        explicit TrackedMem(size_t capacity = CXM_DEFAULT_POOL_SIZE);
        TrackedMem(const TrackedMem&) = delete;
        TrackedMem(TrackedMem&&) = delete;
        ~TrackedMem();

        /**
         * @brief Allocates memory from the arena.
         *
         * @param count     Number of `f32` elements to allocate
         * @param alignment Desired alignment in bytes (default 32)
         * @return Pointer to allocated memory, or `nullptr` if allocation fails
         */
        [[nodiscard]]
        f32* allocate(size_t count, size_t alignment = 32);
        /**
         * @brief Deallocates previously allocated memory.
         *
         * @param ptr Pointer previously returned by `allocate()`
         */
        void deallocate(const f32* ptr);
        /**
         * @brief Resets the entire arena, marking all memory as free.
         */
        void reset();
        /**
         * @brief Returns the total capacity of the arena.
         */
        [[nodiscard]]
        size_t capacity() const noexcept;
        /**
         * @brief Returns the number of currently allocated `f32` elements.
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
         * @brief Merges adjacent free blocks (coalescing).
         */
        void coalesce(std::map<size_t, Arena>::iterator it);
    };
} //namespace cortex::_fw::sys

#endif //CORTEXMIND_FRAMEWORK_MEMORY_MEM_HPP