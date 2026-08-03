//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_FORGE_CUH
#define CORTEXMIND_FRAMEWORK_MEMORY_FORGE_CUH

#include <CortexMind/framework/Memory/arena.hpp>
#include <map>
#include <mutex>

namespace cortex::_fw {
    class ForgeChunk {
    public:
        explicit ForgeChunk(std::size_t bytes);
        ForgeChunk(const ForgeChunk&) = delete;
        ForgeChunk(ForgeChunk&&) = delete;
        ~ForgeChunk();

        [[nodiscard]]
        std::byte* allocate(std::size_t count, std::size_t alignment = 16);
        void deallocate(const std::byte* ptr);
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