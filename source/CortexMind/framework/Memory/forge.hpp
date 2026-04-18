//
// Created by muham on 17.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_FORGE_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_FORGE_HPP

#include <CortexMind/framework/Memory/arena.hpp>
#include <CortexMind/framework/Tools/params.hpp>
#include <CortexMind/runtime/macros.hpp>
#include <map>
#include <mutex>

namespace cortex::_fw::sys {
    /**
     * @brief CUDA-aware memory chunk allocator with arena-based management.
     *
     * Manages a single large GPU buffer (allocated via `cudaMalloc`) and tracks
     * allocated/free regions using arenas. Supports aligned allocation and
     * automatic coalescing of adjacent free blocks to minimize fragmentation.
     *
     * Thread-safe via internal mutex.
     */
    class ForgeChunk {
    public:
        /**
         * @brief Constructs a new memory chunk with the given capacity.
         * @param capacity Total capacity in number of `f32` elements (default: CXM_DEFAULT_POOL_SIZE)
         */
        explicit ForgeChunk(size_t capacity = CXM_DEFAULT_POOL_SIZE);
        ForgeChunk(const ForgeChunk&) = delete;
        ForgeChunk(ForgeChunk&&) = delete;
        ~ForgeChunk();

        /**
         * @brief Allocates memory for the requested number of float elements on the GPU.
         * @param count      Number of `f32` elements to allocate
         * @param alignment  Alignment requirement in bytes (default: 16)
         * @return Device pointer to allocated memory, or `nullptr` if allocation fails
         */
        [[nodiscard]]
        f32* allocate(size_t count, size_t alignment = 16);
        /**
         * @brief Deallocates a previously allocated block.
         * @param ptr Device pointer previously returned by `allocate()`
         */
        void deallocate(const f32* ptr);
        /**
         * @brief Resets the chunk, marking all memory as free again.
         */
        void reset();

        /**
         * @brief Returns the total capacity of this chunk in number of `f32` elements.
         */
        [[nodiscard]]
        size_t capacity() const noexcept;
        /**
         * @brief Returns the number of currently used `f32` elements.
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
         * @param it Iterator to the arena that was just freed
         */
        void coalesce(std::map<size_t, Arena>::iterator it);
    };
} //namespace cortex::_fw::sys

#endif //CORTEXMIND_FRAMEWORK_MEMORY_FORGE_HPP