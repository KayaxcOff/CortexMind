//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_FORGE_CUH
#define CORTEXMIND_FRAMEWORK_MEMORY_FORGE_CUH

#include <CortexMind/framework/Memory/arena.hpp>
#include <map>
#include <mutex>

namespace cortex::_fw {
    /**
     * @brief Arena-based CUDA device memory allocator.
     *
     * ForgeChunk manages a contiguous block of CUDA device memory and
     * provides aligned sub-allocations without repeatedly invoking
     * cudaMalloc() and cudaFree().
     *
     * Memory blocks are tracked internally using @ref Arena structures.
     * Free blocks are automatically split during allocation and merged
     * during deallocation to minimize fragmentation.
     *
     * The allocator is thread-safe and serializes allocation requests
     * using an internal mutex.
     *
     * ForgeChunk is intended for high-frequency GPU memory allocations,
     * such as tensor storage and temporary computation buffers.
     */
    class ForgeChunk {
    public:
        explicit ForgeChunk(std::size_t bytes);
        ForgeChunk(const ForgeChunk&) = delete;
        ForgeChunk(ForgeChunk&&) = delete;
        ~ForgeChunk();

        /**
         * @brief Allocates an aligned block of CUDA device memory.
         *
         * Searches for a suitable free block inside the arena and returns
         * a pointer into the managed CUDA memory region.
         *
         * @param count Number of bytes to allocate.
         * @param alignment Required byte alignment.
         *
         * @return Pointer to the allocated device memory, or nullptr if
         * allocation fails.
         */
        [[nodiscard]]
        std::byte* allocate(std::size_t count, std::size_t alignment = 16);
        /**
         * @brief Releases a previously allocated device memory block.
         *
         * Marks the allocation as free and attempts to merge adjacent
         * free blocks to reduce fragmentation.
         *
         * @param ptr Pointer previously returned by allocate().
         */
        void deallocate(const std::byte* ptr);
        /**
         * @brief Resets the allocator to its initial state.
         *
         * All tracked allocations become invalid and the entire CUDA
         * memory region is marked as free.
         */
        void reset();
        [[nodiscard]]
        std::size_t capacity() const noexcept;
        [[nodiscard]]
        std::size_t used() const noexcept;
    private:
        std::byte* m_buffer;
        std::size_t m_capacity;
        std::size_t m_used;

        std::mutex m_mutex;
        std::map<std::size_t, Arena> m_arenas;

        void coalesce(std::map<std::size_t, Arena>::iterator it);
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_MEMORY_FORGE_CUH