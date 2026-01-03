//
// Created by muham on 29.12.2025.
//

#ifndef CORTEXMIND_CORE_ENGINE_STORAGE_STORAGE_HPP
#define CORTEXMIND_CORE_ENGINE_STORAGE_STORAGE_HPP

#include <core/Engine/Memory/alloc.hpp>

namespace cortex::_fw {
    /// @brief Global allocator for tensor storage using tracked memory.
    inline sys::TrackedMem alloc;

    /// @brief A simple contiguous memory storage for tensor data.
    ///
    /// @details
    /// TensorStorage manages a contiguous block of memory for storing tensor elements.
    /// It supports copy and move semantics. Copying duplicates the underlying data,
    /// while moving transfers ownership without copying. Memory is allocated using
    /// a polymorphic allocator, which defaults to the global tracked allocator.
    struct TensorStorage {
        /// @brief Constructs a TensorStorage of given size.
        /// @param size Number of elements to allocate
        /// @param resource Memory resource to use for allocation (default: tracked global allocator)
        explicit TensorStorage(const std::size_t size, std::pmr::memory_resource* resource = &alloc) : m_size(size), m_alloc(resource) {
            this->m_data = this->m_alloc.allocate(this->m_size);
        }

        /// @brief Copy constructor
        /// @param other TensorStorage to copy from
        /// @details Allocates new memory and copies all elements from the source.
        TensorStorage(const TensorStorage& other) : m_size(other.m_size), m_alloc(other.m_alloc) {
            if (this->m_size > 0) {
                this->m_data = this->m_alloc.allocate(this->m_size);
                std::memcpy(this->m_data, other.m_data, this->m_size * sizeof(float));
            }
        }

        /// @brief Move constructor
        /// @param other TensorStorage to move from
        /// @details Transfers ownership of memory without copying. Source is invalidated.
        TensorStorage(TensorStorage&& other) noexcept : m_data(other.m_data), m_size(other.m_size), m_alloc(other.m_alloc) {
            other.m_data = nullptr;
            other.m_size = 0;
        }

        /// @brief Destructor
        /// @details Deallocates memory if valid.
        ~TensorStorage() {
            if (this->m_data) {
                this->m_alloc.deallocate(this->m_data, this->m_size);
                this->m_data = nullptr;
                this->m_size = 0;
            }
        }

        /// @brief Returns a pointer to the underlying data.
        float* ptr() { return this->m_data; }

        /// @brief Returns a const pointer to the underlying data.
        const float* ptr() const { return this->m_data; }

        /// @brief Returns the number of elements in storage.
        size_t size() const { return this->m_size; }

        /// @brief Checks if the storage is empty.
        bool empty() const { return this->m_size == 0; }

        /// @brief Checks if the storage is valid (allocated).
        bool is_valid() const { return this->m_data != nullptr; }
    private:
        float* m_data;
        size_t m_size;
        std::pmr::polymorphic_allocator<float> m_alloc;
    };
} // namespace cortex::_fw

#endif // CORTEXMIND_CORE_ENGINE_STORAGE_STORAGE_HPP