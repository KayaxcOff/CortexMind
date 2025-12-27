#include "CortexMind/core/Engine/Memory/alloc.hpp"

using namespace cortex::_fw::tiny;

TensorMemoryPool::TensorMemoryPool(std::size_t _pool_size) : pool_size(_pool_size), offset(0) {
	this->pool = ::operator new(this->pool_size, std::align_val_t(alignof(std::max_align_t)));
}

TensorMemoryPool::~TensorMemoryPool() {
	::operator delete(this->pool, std::align_val_t(alignof(std::max_align_t)));
}

void* TensorMemoryPool::do_allocate(std::size_t bytes, std::size_t alignment) {
	std::lock_guard<std::mutex> lock(this->mtx);
	for (auto item = this->free_blocks.begin(); item != this->free_blocks.end(); ++item) {
		if (item->size >= bytes) {
			void* ptr = item->ptr;
			blocks.push_back({ ptr, bytes, true });
			free_blocks.erase(item);
			return ptr;
		}
	}

	std::size_t current_offset = (this->offset + (alignment - 1)) & ~(alignment - 1);
	if (current_offset + bytes > this->pool_size) {
		throw std::bad_alloc();
	}

	void* ptr = static_cast<char*>(this->pool) + current_offset;
	this->blocks.push_back({ ptr, bytes, true });
	this->offset = current_offset + bytes;
	return ptr;
}

void TensorMemoryPool::do_deallocate(void* p, std::size_t bytes, std::size_t alignment) {
	std::lock_guard<std::mutex> lock(this->mtx);

	for (auto& block : this->blocks) {
		if (block.ptr == p && block.in_use) {
			block.in_use = false;
			this->free_blocks.push_back({ p, block.size, false });
			return;
		}
	}
}

bool TensorMemoryPool::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
	return this == &other;
}

void TensorMemoryPool::reset() {
	std::lock_guard<std::mutex> lock(this->mtx);
	this->offset = 0;
	this->blocks.clear();
	this->free_blocks.clear();
}

size_t TensorMemoryPool::get_used_memory() const {
	std::lock_guard<std::mutex> lock(this->mtx);
	return offset;
}

size_t TensorMemoryPool::get_active_allocations() const {
	std::lock_guard<std::mutex> lock(this->mtx);
	size_t count = 0;
	for (const auto& block : blocks) {
		if (block.in_use) count++;
	}
	return count;
}