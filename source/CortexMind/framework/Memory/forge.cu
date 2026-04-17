//
// Created by muham on 17.04.2026.
//

#include "CortexMind/framework/Memory/forge.hpp"
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/memory_utils.hpp>

using namespace cortex::_fw::sys;
using namespace cortex::_fw;

ForgeChunk::ForgeChunk(const size_t capacity) : m_capacity(capacity) {
    CXM_CUDA_ASSERT(cuda::malloc(reinterpret_cast<void**>(&this->m_buffer), capacity * sizeof(f32)), "cortex::_fw::sys::ForgeChunk::ForgeChunk()");
    this->m_arenas.push_back({0, capacity, false});
}

ForgeChunk::~ForgeChunk() {
    CXM_CUDA_ASSERT(cuda::free(this->m_buffer), "cortex::_fw::sys::ForgeChunk::~ForgeChunk()");
}

f32* ForgeChunk::allocate(const size_t count, const size_t alignment) {
    std::lock_guard lock(this->m_mutex);

    size_t align_elems = alignment / sizeof(float);

    for (size_t i = 0; i < m_arenas.size(); i++) {
        auto& a = m_arenas[i];
        if (a.used) {
            continue;
        }

        size_t aligned = align_up(a.offset, align_elems);
        size_t padding = aligned - a.offset;

        if (a.size < padding + count) {
            continue;
        }

        size_t remaining = a.size - padding - count;

        Arena old = a;
        m_arenas.erase(m_arenas.begin() + i);


        if (padding > 0) {
            m_arenas.insert(m_arenas.begin() + i++, {old.offset, padding, false});
        }

        m_arenas.insert(m_arenas.begin() + i++, {aligned, count, true});

        if (remaining > 0) {
            m_arenas.insert(m_arenas.begin() + i, {aligned + count, remaining, false});
        }

        m_used += count;
        return m_buffer + aligned;
    }

    return nullptr;
}

void ForgeChunk::deallocate(const float* ptr) {
    if (!ptr) {
        return;
    }

    std::lock_guard lock(m_mutex);

    size_t offset = ptr - m_buffer;

    for (size_t i = 0; i < m_arenas.size(); i++) {
        if (m_arenas[i].offset == offset) {
            m_arenas[i].used = false;
            m_used -= m_arenas[i].size;

            coalesce(i);
            return;
        }
    }
}

void ForgeChunk::reset() {
    std::lock_guard lock(m_mutex);

    m_arenas.clear();
    m_arenas.push_back({0, m_capacity, false});
    m_used = 0;
}

void ForgeChunk::coalesce(size_t i) {
    if (i > 0 && !m_arenas[i - 1].used) {
        m_arenas[i - 1].size += m_arenas[i].size;
        m_arenas.erase(m_arenas.begin() + i);
        i--;
    }

    if (i + 1 < m_arenas.size() && !m_arenas[i + 1].used) {
        m_arenas[i].size += m_arenas[i + 1].size;
        m_arenas.erase(m_arenas.begin() + i + 1);
    }
}