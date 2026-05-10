//
// Created by muham on 10.05.2026.
//

#include "CortexMind/framework/Memory/forge.cuh"
#include <CortexMind/framework/Tools/alignment.hpp>
#include <CortexMind/framework/Tools/cuda.cuh>
#include <CortexMind/framework/Tools/err.hpp>
#include <xutility>

using namespace cortex::_fw::sys;
using namespace cortex::_fw;

ForgeChunk::ForgeChunk(const size_t capacity) : m_capacity(capacity), m_used(0) {
    cuda::malloc(reinterpret_cast<void**>(&this->m_buffer), capacity * sizeof(f32));
    this->m_arenas.emplace(0, Arena{0, capacity, false});
}

ForgeChunk::~ForgeChunk() {
    cuda::free(this->m_buffer);
}

f32* ForgeChunk::allocate(const size_t count, const size_t alignment) {
    std::lock_guard lock(this->m_mutex);

    const size_t align_elems = alignment / sizeof(f32);

    for (auto it = this->m_arenas.begin(); it != this->m_arenas.end(); ++it) {
        if (it->second.used) {
            continue;
        }

        const size_t raw_offset     = it->second.offset;
        const size_t aligned_offset = align_up(raw_offset, align_elems);
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

void ForgeChunk::deallocate(const f32* ptr) {
    if (!ptr) {
        return;
    }

    std::lock_guard lock(this->m_mutex);

    const size_t offset = ptr - this->m_buffer;

    if (offset >= this->m_capacity) {
        CXM_WARN(true, "Pointer out of bounds!");
        return;
    }

    const auto it = this->m_arenas.find(offset);
    if (it == this->m_arenas.end()) {
        return;
    }

    it->second.used = false;
    this->m_used -= it->second.size;

    this->coalesce(it);
}

void ForgeChunk::reset() {
    std::lock_guard lock(this->m_mutex);

    this->m_arenas.clear();
    this->m_arenas.emplace(0, Arena{0, this->m_capacity, false});
    this->m_used = 0;
}

size_t ForgeChunk::capacity() const noexcept {
    return this->m_capacity;
}

size_t ForgeChunk::used() const noexcept {
    return this->m_used;
}

void ForgeChunk::coalesce(std::map<size_t, Arena>::iterator it) {
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