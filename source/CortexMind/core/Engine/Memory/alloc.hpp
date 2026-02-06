//
// Created by muham on 4.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_MEMORY_ALLOC_HPP
#define CORTEXMIND_CORE_ENGINE_MEMORY_ALLOC_HPP

#include <CortexMind/core/Engine/Memory/util.hpp>
#include <memory_resource>
#include <vector>
#include <mutex>

namespace cortex::_fw::sys {
    /**
     * @brief Polymorphic memory resource with arena-based tracking.
     *
     * TrackedMem is a custom PMR implementation that allocates memory
     * from internally managed arenas. It is designed for engine-level
     * systems where allocation patterns are predictable and frequent
     * deallocations are unnecessary.
     *
     * Deallocation requests are intentionally ignored; memory is
     * reclaimed only via reset() or destruction.
     *
     * Thread safety:
     *  - Allocation is thread-safe.
     *  - Deallocation is a no-op.
     */
    class TrackedMem : public std::pmr::memory_resource {
    public:
        /**
         * @brief Constructs a tracked memory resource.
         *
         * @param _arena_size Size in bytes of each internal arena.
         *                    Defaults to 1 MB.
         */
        explicit TrackedMem(size_t _arena_size = 1024 * 1024);
        /**
         * @brief Destroys the memory resource and releases all arenas.
         */
        ~TrackedMem() override;

        /**
         * @brief Resets all arenas, invalidating existing allocations.
         *
         * After calling reset(), all previously returned pointers
         * become invalid. This operation does not free arena memory
         * but rewinds allocation state.
         */
        void reset();
    protected:
        /**
         * @brief Allocates memory from the internal arenas.
         *
         * @param bytes Number of bytes to allocate.
         * @param alignment Required alignment.
         * @return Pointer to allocated memory.
         *
         * This function is thread-safe and will create new arenas
         * if existing ones cannot satisfy the request.
         */
        void* do_allocate(size_t bytes, size_t alignment) override;
        /**
         * @brief Deallocation is intentionally ignored.
         *
         * Memory is reclaimed only via reset() or destruction.
         */
        void  do_deallocate(void*, size_t, size_t) override {}
        /**
         * @brief Compares two memory resources for equality.
         *
         * @return True if both resources refer to the same instance.
         */
        [[nodiscard]]
        bool  do_is_equal(const memory_resource& other) const noexcept override;
    private:
        std::vector<Arena> arenas;
        std::mutex mutex;
        size_t arena_size;
    };
} // namespace cortex::_fw::sys

#endif //CORTEXMIND_CORE_ENGINE_MEMORY_ALLOC_HPP