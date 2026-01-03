//
// Created by muham on 29.12.2025.
//

#include "core/Engine/Memory/alloc.hpp"

using namespace cortex::_fw::sys;

TrackedMem::TrackedMem(const size_t default_pool_size) : pool_size(default_pool_size) {
    this->pools.push_back({::operator new(this->pool_size), this->pool_size, 0});
}

void *TrackedMem::do_allocate(const size_t bytes, const size_t alignment) {
    std::lock_guard lock(this->mtx);

    for (auto item = this->free_blocks.begin(); item != this->free_blocks.end(); ++item) {
        if (item->size >= bytes) {
            void* ptr = item->ptr;
            item->in_use = true;
            this->blocks.push_back(*item);
            this->free_blocks.erase(item);
            return ptr;
        }
    }

    for (auto& item : this->pools) {
        const size_t currOffset = item.offset;
        size_t algOffset = (currOffset + alignment - 1) & ~(alignment - 1);
        if (algOffset + bytes <= item.size) {
            void* ptr = static_cast<char*>(item.ptr) + algOffset;
            item.offset = algOffset + bytes;
            this->blocks.push_back({ ptr, bytes, true });
            return ptr;
        }
    }

    const size_t newPoolSize = std::max(this->pool_size, std::max(bytes + alignment, this->pool_size * 2));
    this->pools.push_back({ ::operator new(newPoolSize), newPoolSize, 0 });
    PoolBlock& new_pool = this->pools.back();
    const size_t algOffset = (0 + alignment - 1) & ~(alignment - 1);
    void* ptr = static_cast<char*>(new_pool.ptr) + algOffset;
    new_pool.offset = algOffset + bytes;
    this->blocks.push_back({ ptr, bytes, true });
    return ptr;
}

void TrackedMem::do_deallocate(void* p, size_t bytes, size_t alignment) {
    std::lock_guard lock(this->mtx);
    for (auto item = this->blocks.begin(); item != this->blocks.end(); ++item) {
        if (item->ptr == p && item->in_use) {
            item->in_use = false;
            this->free_blocks.push_back(*item);
            this->blocks.erase(item);
            return;
        }
    }
}

bool TrackedMem::do_is_equal(const memory_resource& other) const noexcept {
    return this == &other;
}

void TrackedMem::reset() {
    std::lock_guard lock(this->mtx);
    this->blocks.clear();
    this->free_blocks.clear();
    for (const auto& item : this->pools) ::operator delete(item.ptr, static_cast<std::align_val_t>(alignof(std::max_align_t)));
    this->pools.clear();
}