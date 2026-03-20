//
// Created by muham on 14.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_MEMORY_BUF_CUH
#define CORTEXMIND_CORE_ENGINE_MEMORY_BUF_CUH

#include <CortexMind/core/Engine/CUDA/params.cuh>
#include <CortexMind/core/Engine/Memory/utils.hpp>
#include <mutex>
#include <vector>

namespace cortex::_fw::sys {
    /**
     * @brief   Thread-safe CUDA device memory pool with sub-allocation
     *
     * Allocates large arenas asynchronously from a cudaMemPool, then manages
     * smaller allocations via bump (arena.offset) and best-fit free list.
     */
    class DeviceHeap {
    public:
        /**
         * @brief   Constructor – creates CUDA memory pool and dedicated stream
         * @param   chunk_size   Size of each new arena/chunk (default 256 MB)
         *
         * @note    Uses cudaMemAllocationTypePinned + device location
         * @note    Creates a non-blocking stream for async malloc/free
         * @throws  Throws via CXM_ASSERT on cudaMemPoolCreate / cudaStreamCreate failure
         */
        DeviceHeap(size_t chunk_size = 256ULL * 1024 * 1024);
        /**
         * @brief   Destructor – frees all arenas and destroys pool/stream
         *
         * @note    Synchronizes the internal stream before destroying
         * @note    Clears arenas and free_list
         */
        ~DeviceHeap();

        /**
         * @brief   Allocates aligned device memory from the pool
         * @param   bytes   Requested size in bytes
         * @return  Aligned device pointer (256-byte alignment)
         *
         * @note    Tries bump allocation first (fast path)
         * @note    Falls back to best-fit from free list
         * @note    Adds new arena if necessary
         * @note    Thread-safe (mutex protected)
         */
        void* allocate(size_t bytes);
        /**
         * @brief   Returns memory to the free list (does not call cudaFreeAsync immediately)
         * @param   ptr     Pointer previously returned by allocate()
         * @param   bytes   Original requested size
         *
         * @note    Aligns size to 256 bytes and pushes to free_list
         * @note    No immediate cudaFreeAsync – deferred until arena release
         * @note    Thread-safe
         */
        void  deallocate(void* ptr, size_t bytes);
    private:
        cudaMemPool_t pool;
        cudaStream_t stream;
        std::vector<Arena> arenas;
        std::vector<Block> free_list;
        size_t chunk_size;
        std::mutex mtx;

        /**
         * @brief   Adds a new large arena using cudaMallocFromPoolAsync
         *
         * @note    Allocates chunk_size bytes asynchronously
         * @note    Synchronizes the pool stream to make pointer visible
         */
        void add_arena();
        /**
         * @brief   Releases all arenas (async free + sync)
         *
         * @note    Called in destructor
         * @note    Frees via cudaFreeAsync on pool stream
         * @note    Synchronizes before clearing containers
         */
        void release();
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

#endif //CORTEXMIND_CORE_ENGINE_MEMORY_BUF_CUH