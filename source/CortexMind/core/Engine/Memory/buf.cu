//
// Created by muham on 14.03.2026.
//

#include "CortexMind/core/Engine/Memory/buf.cuh"
#include <CortexMind/core/Tools/err.hpp>
#include <algorithm>

using namespace cortex::_fw::sys;

DeviceHeap::DeviceHeap(const size_t chunk_size) : chunk_size(chunk_size) {
    cudaMemPoolProps props{};
    props.allocType   = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    cudaGetDevice(&props.location.id);

    CXM_ASSERT(cudaMemPoolCreate(&this->pool, &props) == cudaSuccess,
        "cortex::_fw::sys::DeviceHeap::DeviceHeap()", "Failed to create CUDA memory pool");
    CXM_ASSERT(cudaStreamCreate(&this->stream) == cudaSuccess,
        "cortex::_fw::sys::DeviceHeap::DeviceHeap()", "Failed to create CUDA stream");

    this->add_arena();
}

DeviceHeap::~DeviceHeap() {
    this->release();
    cudaStreamDestroy(this->stream);
    cudaMemPoolDestroy(this->pool);
}

void* DeviceHeap::allocate(const size_t bytes) {
    std::lock_guard lock(this->mtx);

    constexpr size_t ALIGN = 256;
    const size_t aligned_bytes = align_up(bytes, ALIGN);

    for (auto item = this->free_list.begin(); item != this->free_list.end(); ++item) {
        if (item->size == aligned_bytes) {
            void* ptr = item->ptr;
            this->free_list.erase(item);
            return ptr;
        }
    }

    auto best_item = this->free_list.end();
    for (auto item = this->free_list.begin(); item != this->free_list.end(); ++item) {
        if (item->size >= aligned_bytes) {
            if (best_item == this->free_list.end() || item->size < best_item->size)
                best_item = item;
        }
    }
    if (best_item != this->free_list.end()) {
        void* ptr = best_item->ptr;
        this->free_list.erase(best_item);
        return ptr;
    }

    for (auto& item : this->arenas) {
        const size_t aligned_offset = align_up(item.offset, ALIGN);
        if (aligned_offset + aligned_bytes <= item.size) {
            void* ptr    = static_cast<char*>(item.ptr) + aligned_offset;
            item.offset = aligned_offset + aligned_bytes;
            return ptr;
        }
    }

    this->add_arena();
    return this->allocate(bytes);
}

void DeviceHeap::deallocate(void* ptr, const size_t bytes) {
    std::lock_guard lock(this->mtx);
    constexpr size_t ALIGN = 256;
    this->free_list.push_back({ptr, align_up(bytes, ALIGN)});
    if (this->free_list.size() >= 64) {
        this->coalesce();
    }
}

void DeviceHeap::add_arena() {
    void* ptr = nullptr;
    CXM_ASSERT(cudaMallocFromPoolAsync(&ptr, this->chunk_size, this->pool, this->stream) == cudaSuccess,
        "DeviceHeap::add_arena()", "Failed to allocate arena from CUDA pool");
    CXM_ASSERT(cudaStreamSynchronize(this->stream) == cudaSuccess,
        "DeviceHeap::add_arena()", "Failed to synchronize stream");
    this->arenas.push_back({ptr, this->chunk_size, 0});
}

void DeviceHeap::release() {
    for (const auto& item : this->arenas) {
        cudaFreeAsync(item.ptr, this->stream);
    }
    cudaStreamSynchronize(this->stream);
    this->arenas.clear();
    this->free_list.clear();
}

void DeviceHeap::coalesce() {
    std::sort(this->free_list.begin(), this->free_list.end(), [](const Block& a, const Block& b) { return a.ptr < b.ptr; });

    std::vector<Block> new_list;
    if (!this->free_list.empty()) {
        Block current = this->free_list[0];
        for (size_t i = 1; i < this->free_list.size(); ++i) {
            char* next_start = static_cast<char*>(this->free_list[i].ptr);
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