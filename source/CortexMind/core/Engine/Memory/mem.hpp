//
// Created by muham on 22.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_MEMORY_MEM_HPP
#define CORTEXMIND_CORE_ENGINE_MEMORY_MEM_HPP

#include <CortexMind/core/Engine/Memory/arena.hpp>
#include <memory_resource>
#include <vector>
#include <mutex>

namespace cortex::_fw::sys {
    class TrackedMem : public std::pmr::memory_resource {
    public:
        explicit
        TrackedMem(size_t _arena_size = 1024 * 1024);
        ~TrackedMem() override;

        void reset();
    protected:
        void* do_allocate(size_t bytes, size_t alignment) override;
        void  do_deallocate(void*, size_t, size_t) override {}
        [[nodiscard]]
        bool  do_is_equal(const memory_resource& other) const noexcept override;
    private:
        std::vector<Arena> arenas;
        std::mutex mutex;
        size_t arena_size;
    };
} // namespace cortex::_fw::sys

#endif //CORTEXMIND_CORE_ENGINE_MEMORY_MEM_HPP