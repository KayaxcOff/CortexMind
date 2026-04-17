//
// Created by muham on 17.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_FORGE_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_FORGE_HPP

#include <CortexMind/framework/Memory/arena.hpp>
#include <CortexMind/framework/Tools/params.hpp>
#include <CortexMind/runtime/macros.hpp>
#include <mutex>
#include <vector>

namespace cortex::_fw::sys {
    class ForgeChunk {
    public:
        explicit ForgeChunk(size_t capacity = CXM_DEFAULT_POOL_SIZE);
        ForgeChunk(const ForgeChunk&) = delete;
        ForgeChunk(ForgeChunk&&) = delete;
        ~ForgeChunk();

        [[nodiscard]]
        f32* allocate(size_t count, size_t alignment = 32);
        void deallocate(const f32* ptr);
        void reset();
        [[nodiscard]]
        size_t capacity() const noexcept;
        [[nodiscard]]
        size_t used() const noexcept;
    private:
        f32* m_buffer;
        size_t m_capacity;
        size_t m_used;

        std::mutex m_mutex;
        std::vector<Arena> m_arenas;

        void coalesce(size_t index);
    };
} //namespace cortex::_fw::sys

#endif //CORTEXMIND_FRAMEWORK_MEMORY_FORGE_HPP