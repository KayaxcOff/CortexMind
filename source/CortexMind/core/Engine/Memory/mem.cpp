//
// Created by muham on 14.03.2026.
//

#include "CortexMind/core/Engine/Memory/mem.hpp"
#include <CortexMind/core/Tools/err.hpp>
#include <algorithm>

using namespace cortex::_fw::sys;

TrackedMem::TrackedMem(const size_t arena_size) : arena_size(arena_size) {
    this->add_arena(this->arena_size);
}

TrackedMem::~TrackedMem() {
    for (const auto& item : this->arenas) {
        ::operator delete(item.ptr, static_cast<std::align_val_t>(32));
    }
    this->arenas.clear();
    this->free_list.clear();
}

void *TrackedMem::allocate(const size_t bytes, const size_t alignment) {
    std::lock_guard lock(this->mtx);
    const size_t aligned_bytes = align_up(bytes, alignment);

    for (auto item = this->free_list.begin(); item != this->free_list.end(); ++item) {
        if (item->size == aligned_bytes) {
            void* ptr = item->ptr;
            this->free_list.erase(item);
            return ptr;
        }
    }

    auto best_item = this->free_list.begin();
    for (auto item = this->free_list.begin(); item != this->free_list.end(); ++item) {
        if (item->size >= aligned_bytes) {
            if (best_item == this->free_list.end() || item->size < best_item->size) {
                best_item = item;
            }
        }
    }
    if (best_item != this->free_list.end()) {
        void* ptr = best_item->ptr;
        this->free_list.erase(best_item);
        return ptr;
    }

    this->coalesce();
    for (auto item = this->free_list.begin(); item != this->free_list.end(); ++item) {
        if (item->size >= aligned_bytes) {
            void* ptr = item->ptr;
            this->free_list.erase(item);
            return ptr;
        }
    }

    for (auto& item : this->arenas) {
        const size_t aligned_offset = align_up(item.offset, alignment);
        if (aligned_offset + aligned_bytes <= item.size) {
            void* ptr    = static_cast<char*>(item.ptr) + aligned_offset;
            item.offset = aligned_offset + aligned_bytes;
            return ptr;
        }
    }

    this->add_arena(aligned_bytes);
    return this->allocate(bytes, alignment);
}

void TrackedMem::deallocate(void *ptr, const size_t bytes) {
    std::lock_guard lock(this->mtx);
    this->free_list.push_back({ptr, align_up(bytes, 32)});
    if (this->free_list.size() >= 64) {
        this->coalesce();
    }
}

void TrackedMem::reset() {
    std::lock_guard lock(this->mtx);
    for (auto& item : this->arenas) item.offset = 0;
    this->free_list.clear();
}

void TrackedMem::add_arena(const size_t min_size) {
    const size_t size = std::max(this->arena_size, min_size);
    void* ptr = ::operator new(size, static_cast<std::align_val_t>(32));
    CXM_ASSERT(ptr != nullptr, "cortex::_fw::sys::TrackedMem::add_arena()", "Failed to allocate arena");
    this->arenas.push_back({ptr, size, 0});
}

void TrackedMem::coalesce() {
    std::ranges::sort(this->free_list, [](const Block& a, const Block& b) { return a.ptr < b.ptr; });

    std::vector<Block> new_list;
    if (!this->free_list.empty()) {
        Block current = this->free_list[0];
        for (size_t i = 1; i < this->free_list.size(); ++i) {
            const auto next_start = static_cast<char*>(this->free_list[i].ptr);
            char* curr_end   = static_cast<char*>(current.ptr) + current.size;
            if (next_start == curr_end) {
                current.size += this->free_list[i].size;
            } else {
                new_list.push_back(current);
                current = this->free_list[i];
            }
        }
        new_list.push_back(current);
    }
    this->free_list = std::move(new_list);
}