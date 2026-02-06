//
// Created by muham on 4.02.2026.
//

#include "CortexMind/core/Engine/Memory/alloc.hpp"
#include <algorithm>

using namespace cortex::_fw::sys;

static size_t align_up(const size_t value, const size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

TrackedMem::TrackedMem(const size_t _arena_size) : arena_size(_arena_size) {
    const size_t size = std::max(this->arena_size, this->arena_size);

    void* ptr = ::operator new(size, static_cast<std::align_val_t>(alignof(std::max_align_t)));

    this->arenas.push_back({ptr, size, 0});
}

TrackedMem::~TrackedMem() {
    this->reset();
}

void *TrackedMem::do_allocate(const size_t bytes, const size_t alignment) {
    std::lock_guard lock(this->mutex);

    for (auto& item : this->arenas) {
        size_t aligned = align_up(bytes, alignment);

        if (aligned + bytes <= item.size) {
            void* ptr = static_cast<char*>(item.ptr) + aligned;
            item.offset += aligned + bytes;
            return ptr;
        }
    }

    const size_t size = std::max(this->arena_size, bytes + alignment);
    void* ptr = ::operator new(size, static_cast<std::align_val_t>(alignof(std::max_align_t)));
    this->arenas.push_back({ptr, size, 0});

    return do_allocate(bytes, alignment);
}

bool TrackedMem::do_is_equal(const memory_resource& other) const noexcept {
    return this == &other;
}

void TrackedMem::reset() {
    std::lock_guard lock(this->mutex);

    for (auto& item : this->arenas) {
        ::operator delete(
            item.ptr,
            static_cast<std::align_val_t>(alignof(std::max_align_t))
        );
    }
    this->arenas.clear();
}