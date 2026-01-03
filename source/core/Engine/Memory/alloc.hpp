//
// Created by muham on 29.12.2025.
//

#ifndef CORTEXMIND_ENGINE_MEMORY_ALLOC_HPP
#define CORTEXMIND_ENGINE_MEMORY_ALLOC_HPP

#include <core/Engine/Memory/blocks.hpp>

#include <memory_resource>
#include <vector>
#include <mutex>

namespace cortex::_fw::sys {
    /// @brief A memory resource that tracks allocations and manages memory pools
    ///
    /// @details
    /// TrackedMem inherits from std::pmr::memory_resource and provides a custom
    /// allocator that keeps track of allocated memory blocks. It is thread-safe
    /// via a mutex, supports allocation/deallocation, and can reset all pools
    class TrackedMem : public std::pmr::memory_resource {
    public:
        /// @brief Constructs a TrackedMem with a default pool size.
        /// @param default_pool_size Size of each memory pool in bytes (default: 1 MB)
        explicit TrackedMem(size_t default_pool_size = 1024 * 1024);
        ~TrackedMem() override = default;

        /// @brief Resets all memory pools, effectively freeing all allocation
        void reset();
    protected:
        /// @brief Allocates memory of given size and alignment
        /// @param bytes Number of bytes to allocate
        /// @param alignment Required alignment
        /// @return Pointer to the allocated memory
        void* do_allocate(size_t bytes, size_t alignment) override;

        /// @brief Deallocates previously allocated memory
        /// @param p Pointer to the memory to deallocate
        /// @param bytes Number of bytes that were allocated
        /// @param alignment Alignment that was used during allocation
        void do_deallocate(void* p, size_t bytes, size_t alignment) override;

        /// @brief Compares this memory resource with another for equality
        /// @param other Another memory_resource to compare with
        /// @return true if they are the same resource
        bool do_is_equal(const memory_resource& other) const noexcept override;
    private:
        std::vector<PoolBlock> pools;
        std::vector<Block> blocks;
        std::vector<Block> free_blocks;
        std::mutex mtx;
        size_t pool_size;
    };
} // namespace cortex::_fw::sys

#endif // CORTEXMIND_ENGINE_MEMORY_ALLOC_HPP