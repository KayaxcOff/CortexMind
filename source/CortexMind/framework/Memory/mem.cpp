//
// Created by muham on 17.04.2026.
//

#include "CortexMind/framework/Memory/mem.hpp"
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/memory_utils.hpp>
#include <new>

using namespace cortex::_fw::sys;
using namespace cortex::_fw;

TrackedMem::TrackedMem(const size_t capacity) : m_capacity(capacity), m_used(0) {
    this->m_buffer = static_cast<f32*>(operator new(capacity * sizeof(f32), static_cast<std::align_val_t>(alignof(f32))));
    CXM_ASSERT(!this->m_buffer, "cortex::_fw::sys::TrackedMem::TrackedMem()", "Buffer initialize of buffer has failed");
}

TrackedMem::~TrackedMem() {
    operator delete(m_buffer, static_cast<std::align_val_t>(alignof(f32)));
}

f32 *TrackedMem::allocate(const size_t count, const size_t alignment) {
    std::lock_guard lock(this->m_mutex);

    for (auto item = this->m_arenas.begin(); item != this->m_arenas.end(); ++item) {
        Arena& arena = item->second;

        if (arena.used) {
            continue;
        }

        const size_t raw_offset = arena.offset;
        const size_t aligned_offset = align_up(raw_offset, alignment / sizeof(f32));

        const size_t padding = aligned_offset - raw_offset;

        if (arena.size < padding + count) {
            continue;
        }

        if (padding > 0) {
            Arena prefix{raw_offset, padding, false};
            item = this->m_arenas.emplace(raw_offset, prefix).first;

            arena.offset += padding;
            arena.size -= padding;
        }

        const size_t remaining = arena.size - count;

        const Arena allocated{arena.offset, count, true};
        this->m_arenas[allocated.offset] = allocated;

        if (remaining > 0) {
            const Arena suffix{arena.offset + count, remaining, false};
            this->m_arenas[suffix.offset] = suffix;
        }

        this->m_arenas.erase(item);
        this->m_used += count;

        return this->m_buffer + raw_offset;
    }
    return nullptr;
}

void TrackedMem::deallocate(const f32 *ptr) {
    if (!ptr) {
        return;
    }

    std::lock_guard lock(this->m_mutex);
    const size_t offset = ptr - this->m_buffer;

    const auto item = this->m_arenas.find(offset);
    if (item == this->m_arenas.end()) {
        return;
    }
    item->second.used = false;
    this->m_used += item->second.size;

    this->coalesce(item);
}

void TrackedMem::reset() {
    std::lock_guard lock(this->m_mutex);

    this->m_arenas.clear();
    this->m_arenas.emplace(0, Arena{0, this->m_capacity, false});
    this->m_used = 0;
}

size_t TrackedMem::capacity() const noexcept {
    return this->m_capacity;
}

size_t TrackedMem::used() const noexcept {
    return this->m_used;
}

void TrackedMem::coalesce(std::map<size_t, Arena>::iterator it) {
    if (it != this->m_arenas.begin()) {
        auto prev = std::prev(it);
        if (!prev->second.used) {
            prev->second.size += it->second.size;
            this->m_arenas.erase(it);
            it = prev;
        }
    }

    auto next = std::next(it);
    if (next != this->m_arenas.end() && !next->second.used) {
        it->second.size += next->second.size;
        this->m_arenas.erase(next);
    }
}