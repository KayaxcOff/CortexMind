#ifndef CORTEXMIND_CORE_ENGINE_MEMORY_ALLOC_HPP
#define CORTEXMIND_CORE_ENGINE_MEMORY_ALLOC_HPP

#include <memory_resource>
#include <mutex>
#include <vector>

namespace cortex::_fw::tiny {
	struct MemoryBlock {
		void* ptr;
		std::size_t size;
		bool in_use;
	};

	class TensorMemoryPool : public std::pmr::memory_resource {
	public:
		explicit TensorMemoryPool(std::size_t _pool_size);
		~TensorMemoryPool() override;

		void reset();
		size_t get_used_memory() const;
		size_t get_active_allocations() const;

	protected:
		void* do_allocate(std::size_t bytes, std::size_t alignment) override;
		void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override;
		bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;

	private:
		void* pool;
		std::size_t pool_size;
		std::size_t offset;
		std::vector<MemoryBlock> blocks;
		std::vector<MemoryBlock> free_blocks;
		std::mutex mtx;
	};
} // namespace cortex::_fw::tiny

#endif // CORTEXMIND_CORE_ENGINE_MEMORY_ALLOC_HPP