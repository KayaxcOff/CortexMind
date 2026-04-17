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
    CXM_ASSERT(this->m_buffer != nullptr, "cortex::_fw::sys::TrackedMem::TrackedMem()", "Buffer initialize of buffer has failed");

    this->m_arenas.emplace(0, Arena{0, capacity, false});
}

TrackedMem::~TrackedMem() {
    operator delete(this->m_buffer, static_cast<std::align_val_t>(alignof(f32)));
}

f32 *TrackedMem::allocate(const size_t count, const size_t alignment) {
    std::lock_guard lock(this->m_mutex);

    for (auto it = this->m_arenas.begin(); it != this->m_arenas.end(); ++it) {
        if (it->second.used) {
            continue;
        }

        const size_t raw_offset     = it->second.offset;
        const size_t aligned_offset = align_up(raw_offset, alignment / sizeof(f32));
        const size_t padding        = aligned_offset - raw_offset;

        if (it->second.size < padding + count) {
            continue;
        }

        const size_t remaining = it->second.size - padding - count;

        this->m_arenas.erase(it);

        if (padding > 0) {
            this->m_arenas.emplace(raw_offset, Arena{raw_offset, padding, false});
        }

        this->m_arenas.emplace(aligned_offset, Arena{aligned_offset, count, true});

        if (remaining > 0) {
            this->m_arenas.emplace(aligned_offset + count, Arena{aligned_offset + count, remaining, false});
        }

        this->m_used += count;
        return this->m_buffer + aligned_offset;
    }
    return nullptr;
}

void TrackedMem::deallocate(const f32 *ptr) {
    if (!ptr) {
        return;
    }

    std::lock_guard lock(this->m_mutex);
    const size_t offset = ptr - this->m_buffer;

    if (offset >= this->m_capacity) {
        CXM_WARN(false, "TrackedMem::deallocate", "Pointer out of bounds!");
        return;
    }

    const auto item = this->m_arenas.find(offset);
    if (item == this->m_arenas.end()) {
        return;
    }
    item->second.used = false;
    this->m_used -= item->second.size;

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