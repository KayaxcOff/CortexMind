//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_MEM_HPP
#define CORTEXMIND_MEM_HPP

#include <CortexMind/framework/Memory/arena.hpp>
#include <cstddef>
#include <map>
#include <mutex>

namespace cortex::_fw {
    class TrackedMem {
    public:
        explicit TrackedMem(std::size_t bytes);
        TrackedMem(const TrackedMem &) = delete;
        TrackedMem(TrackedMem &&) = delete;
        ~TrackedMem();

        [[nodiscard]]
        std::byte* allocate(std::size_t count, std::size_t alignment = 32);
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

#endif //CORTEXMIND_MEM_HPP