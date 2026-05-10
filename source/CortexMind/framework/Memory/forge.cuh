//
// Created by muham on 10.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_FORGE_CUH
#define CORTEXMIND_FRAMEWORK_MEMORY_FORGE_CUH

#include <CortexMind/framework/Memory/arena.hpp>
#include <CortexMind/framework/Tools/types.hpp>
#include <CortexMind/runtime/macros.hpp>
#include <map>
#include <mutex>

namespace cortex::_fw::sys {
    /**
     * @brief Arena-based CUDA device memory allocator with coalescing.
     *
     * `ForgeChunk` is a custom memory manager that allocates a large contiguous
     * block of CUDA device memory and manages smaller allocations within it
     * using an arena (linear allocator) strategy.
     *
     * Features:
     * - Alignment-aware allocations
     * - Automatic coalescing of freed adjacent blocks
     * - Thread-safe operations via mutex
     * - Efficient for repeated allocations in training/inference loops
     */
    class ForgeChunk {
    public:
        /**
         * @brief Constructs a new CUDA memory arena.
         *
         * @param capacity Total capacity in number of `f32` elements.
         */
        explicit ForgeChunk(size_t capacity = CXM_DEFAULT_POOL_SIZE);
        ForgeChunk(const ForgeChunk&) = delete;
        ForgeChunk(ForgeChunk&&) = delete;
        ~ForgeChunk();

        /**
         * @brief Allocates memory from the CUDA arena.
         *
         * @param count     Number of `f32` elements to allocate
         * @param alignment Desired alignment in bytes (default: 16)
         * @return Pointer to allocated device memory, or `nullptr` if failed
         */
        [[nodiscard]]
        f32* allocate(size_t count, size_t alignment = 16);
        /**
         * @brief Deallocates previously allocated memory.
         *
         * @param ptr Pointer previously returned by `allocate()`
         */
        void deallocate(const f32* ptr);
        /**
         * @brief Resets the arena, marking all memory as free.
         */
        void reset();
        /**
         * @brief Returns the total capacity of the arena in `f32` elements.
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

#endif //CORTEXMIND_FRAMEWORK_MEMORY_FORGE_CUH