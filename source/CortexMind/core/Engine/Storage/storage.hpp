#ifndef CORTEXMIND_CORE_ENGINE_STORAGE_STORAGE_HPP
#define CORTEXMIND_CORE_ENGINE_STORAGE_STORAGE_HPP

#include <CortexMind/core/Engine/Memory/alloc.hpp>

/*
One of the main problems related to speed in tensor operations is speed itself. 
To perform operations quickly, memory allocation is necessary because requesting 
memory from the operating system during the operation will slow down the function's 
execution time. TensorStorage exists to prevent tensor functions from returning copy; 
if tensor functions return copy, this will cause significant memory usage.
*/

namespace cortex::_fw {
	inline tiny::TensorMemoryPool alloc(1024 * 1024); /// 1 MB pool allocator

	// @brief Tensor storage class that manages a contiguous block of memory for tensor data
	// @note Uses a polymorphic allocator for memory management
	struct TensorStorage {
		using allocator_type = std::pmr::polymorphic_allocator<float>;

		TensorStorage(std::size_t size,std::pmr::memory_resource* resource = &alloc) : m_size(size), m_alloc(resource) {
			m_data = m_alloc.allocate(m_size);
		}

		TensorStorage(const TensorStorage& other) : m_size(other.m_size), m_alloc(other.m_alloc) {
			if (m_size > 0) {
				m_data = m_alloc.allocate(m_size);
				std::memcpy(m_data, other.m_data, m_size * sizeof(float));
			}
		}

		TensorStorage(TensorStorage&& other) noexcept : m_data(other.m_data), m_size(other.m_size), m_alloc(other.m_alloc) {
			other.m_data = nullptr;
			other.m_size = 0;
		}

		~TensorStorage() {
			if (m_data) {
				m_alloc.deallocate(m_data, m_size);
				m_data = nullptr;
				m_size = 0;
			}
		}

		// @brief Access the underlying data pointer
		float* ptr()  noexcept {return m_data;}

		// @brief Access the underlying data pointer (const version)
		const float* ptr() const noexcept {return m_data;}

		// @brief Get the size of the storage
		std::size_t size() const {return m_size;}

		// @brief Check if the storage is empty
		bool empty() const {return m_size == 0;}
	private:
		float* m_data = nullptr;
		std::size_t    m_size = 0;
		allocator_type m_alloc;
	};
} // namespace cortex::_fw

#endif // CORTEXMIND_CORE_ENGINE_STORAGE_STORAGE_HPP