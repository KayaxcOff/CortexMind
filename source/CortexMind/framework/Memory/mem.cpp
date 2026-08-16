//
// Created by muham on 3.08.2026.
//

#include "CortexMind/framework/Memory/mem.hpp"
#include <CortexMind/framework/Tools/Error/errors.hpp>
#include <tlx/memory.hpp>
#include <xutility>

using namespace cortex::_fw;

constexpr std::size_t kAlignment = 32;

TrackedMem::TrackedMem(const std::size_t bytes) {
    this->m_capacity = bytes;
    this->m_used = 0;

    this->m_buffer = static_cast<std::byte *>(tlx::malloc(bytes, kAlignment));
    CXM_ASSERT( this->m_buffer == nullptr, "Buffer is nullptr");
}

TrackedMem::~TrackedMem() {
    tlx::free(this->m_buffer, kAlignment);
}

std::byte *TrackedMem::allocate(const std::size_t count, const std::size_t alignment) {
    std::lock_guard lock(this->m_mutex);

    for (auto it = this->m_arenas.begin(); it != this->m_arenas.end(); ++it) {
        if (it->second.used) {
            continue;
        }

        std::size_t rawOffset = it->second.offset;
        std::size_t alignedOffset = tlx::alignUp(rawOffset, alignment);
        const std::size_t padding = alignedOffset - rawOffset;

        if (it->second.size < padding + count) {
            continue;
        }

        const std::size_t remaining = it->second.size - padding - count;

        this->m_arenas.erase(it);

        if (padding > 0) {
            this->m_arenas.emplace(rawOffset, Arena{rawOffset, padding, false});
        }

        this->m_arenas.emplace(alignedOffset, Arena{alignedOffset, count, true});

        if (remaining > 0) {
            this->m_arenas.emplace(alignedOffset + count, Arena{alignedOffset + count, remaining, false});
        }

        this->m_used += count;
        return this->m_buffer + alignedOffset;
    }
    return nullptr;
}

void TrackedMem::deallocate(const std::byte *ptr) {
    if (!ptr) {
        return;
    }

    std::lock_guard lock(this->m_mutex);
    const size_t offset = ptr - this->m_buffer;

    if (offset >= this->m_capacity) {
        CXM_WARN(true, "Pointer out of bounds!");
        return;
    }

    const auto item = this->m_arenas.find(offset);
    if (item == this->m_arenas.end()) {
        return;
    }
    item->second.used = false;
    this->m_used -= item->second.size;

    coalesce(item);
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