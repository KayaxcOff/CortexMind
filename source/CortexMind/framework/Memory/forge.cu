//
// Created by muham on 3.08.2026.
//

#include "CortexMind/framework/Memory/forge.cuh"
#include <CortexMind/framework/Tools/Error/errors.hpp>
#include <tlx/memory.hpp>
#include <cuda_runtime.h>
#include <xutility>

using namespace cortex::_fw;

ForgeChunk::ForgeChunk(const std::size_t bytes) {
    this->m_buffer = nullptr;
    this->m_capacity = bytes;
    this->m_used = 0;

    CXM_DEVICE_ASSERT(
        cudaMalloc(reinterpret_cast<void**>(&this->m_buffer), this->m_capacity),
        "Buffer allocation failed"
    );
}

ForgeChunk::~ForgeChunk() {
    cudaFree(this->m_buffer);
}

std::byte *ForgeChunk::allocate(const std::size_t count, const std::size_t alignment) {
    std::lock_guard lock(this->m_mutex);

    const std::size_t align_elems = alignment;

    for (auto it = this->m_arenas.begin(); it != this->m_arenas.end(); ++it) {
        if (it->second.used) {
            continue;
        }

        const std::size_t raw_offset     = it->second.offset;
        const std::size_t aligned_offset = tlx::alignUp(raw_offset, align_elems);
        const std::size_t padding        = aligned_offset - raw_offset;

        if (it->second.size < padding + count) {
            continue;
        }

        const std::size_t remaining = it->second.size - padding - count;

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

void ForgeChunk::deallocate(const std::byte *ptr) {
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

    coalesce(it);
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