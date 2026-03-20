//
// Created by muham on 14.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_MEMORY_MEM_HPP
#define CORTEXMIND_CORE_ENGINE_MEMORY_MEM_HPP

#include <CortexMind/core/Engine/Memory/utils.hpp>
#include <mutex>
#include <vector>

namespace cortex::_fw::sys {
    /**
     * @brief   Thread-safe host memory pool with sub-allocation (CPU-side)
     *
     * Manages large memory arenas allocated via aligned operator new, then sub-allocates
     * smaller blocks using bump allocation (in arenas) and best-fit free list with coalescing.
     *
     * Main goals:
     *   - Minimize frequent ::operator new / delete overhead
     *   - Provide configurable alignment (default 32 bytes, suitable for SIMD/AVX2)
     *   - Thread-safe allocation/deallocation
     *   - Support reset() to reuse arenas without freeing
     *
     * @note    This is a hybrid bump allocator + best-fit free list:
     *            - Fast path: bump allocation in current arena
     *            - Fallback: best-fit from free list
     *            - Coalescing triggered when free_list grows beyond threshold (64 blocks)
     * @note    All allocations are aligned (default 32 bytes, can be overridden)
     * @note    Destruction frees all arenas (no deferred free like DeviceHeap)
     * @note    reset() clears free_list and resets arena offsets (memory reused)
     */
    class TrackedMem {
    public:
        /**
         * @brief   Constructor – allocates initial arena
         * @param   arena_size   Default size of each new arena (default: 256 MiB)
         *
         * @note    Initial arena allocated immediately
         * @note    All arenas use 32-byte alignment by default
         */
        explicit
        TrackedMem(size_t arena_size = 256ULL * 1024 * 1024);
        TrackedMem(const TrackedMem&)            = delete;
        TrackedMem& operator=(const TrackedMem&) = delete;
        TrackedMem(TrackedMem&&)                 = delete;
        TrackedMem& operator=(TrackedMem&&)      = delete;
        /**
         * @brief   Destructor – frees all allocated arenas
         *
         * @note    Calls ::operator delete with alignment on each arena
         * @note    Clears arenas and free_list
         */
        ~TrackedMem();

        /**
         * @brief   Allocates aligned memory from the pool
         * @param   bytes       Requested size in bytes
         * @param   alignment   Desired alignment (default: 32 bytes)
         * @return  Aligned host pointer
         *
         * @note    Fast path: bump allocation in current arena
         * @note    Fallback 1: exact/best-fit from free list
         * @note    Fallback 2: trigger coalescing if free_list large
         * @note    Fallback 3: add new arena if needed (recursive)
         * @note    Thread-safe via mutex
         */
        void* allocate(size_t bytes, size_t alignment = 32);
        /**
         * @brief   Returns memory to the free list
         * @param   ptr     Pointer previously returned by allocate()
         * @param   bytes   Original requested size
         *
         * @note    Aligns size to 32 bytes and adds to free_list
         * @note    Triggers coalescing if free_list grows beyond threshold
         * @note    Thread-safe
         */
        void  deallocate(void* ptr, size_t bytes);
        /**
         * @brief   Resets pool state without freeing memory
         *
         * @note    Sets all arena offsets to 0
         * @note    Clears free_list
         * @note    Allows reusing existing arenas for new allocations
         * @note    Thread-safe
         */
        void  reset();
    private:
        std::vector<Arena> arenas;
        std::vector<Block> free_list;
        std::mutex mtx;
        size_t arena_size;

        /**
         * @brief   Adds a new arena large enough for the request
         * @param   min_size   Minimum size required for this allocation
         *
         * @note    Actual size = max(arena_size, min_size)
         * @note    Uses aligned operator new (32-byte alignment)
         * @note    Throws via CXM_ASSERT on allocation failure
         */
        void add_arena(size_t min_size);
        /**
         * @brief   Merges adjacent free blocks (coalescing)
         *
         * @note    Sorts free_list by address (O(N log N))
         * @note    Scans linearly and merges contiguous blocks
         * @note    Replaces old free_list with merged result
         * @note    Called only when free_list grows beyond threshold
         */
        void coalesce();
    };
} // namespace cortex::_fw::sys

#endif //CORTEXMIND_CORE_ENGINE_MEMORY_MEM_HPP