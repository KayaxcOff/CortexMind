//
// Created by muham on 4.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_STORAGE_STOR_HPP
#define CORTEXMIND_CORE_ENGINE_STORAGE_STOR_HPP

#include <CortexMind/core/Engine/Memory/alloc.hpp>
#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw {
    // Global tracked memory resource used as default allocator.
    inline sys::TrackedMem mem;
    /**
     * @brief PMR-based storage container for tensor data.
     *
     * TensorStorage manages a contiguous block of float data using
     * a polymorphic memory resource. It supports copy and move semantics
     * and does not own the memory resource itself.
     */
    struct TensorStorage {
        /**
         * @brief Constructs a tensor storage with the given size.
         * @param size Number of float elements to allocate.
         * @param resource Memory resource used for allocation.
         */
        explicit TensorStorage(size_t size, std::pmr::memory_resource* resource = &mem);

        /// Copy constructor (deep copy).
        TensorStorage(const TensorStorage& other);
        /// Move constructor.
        TensorStorage(TensorStorage&& other) noexcept;
        /// Releases allocated memory.
        ~TensorStorage();

        /**
         * @brief Returns a pointer to mutable tensor data.
         * @return Pointer to the underlying float buffer.
         */
        f32* data();
        [[nodiscard]]
        /**
         * @brief Returns a pointer to immutable tensor data.
         * @return Pointer to the underlying float buffer.
         */
        const f32* data() const;
        /**
         * @brief Returns the number of stored elements.
         */
        [[nodiscard]]
        size_t size() const;
        /**
        * @brief Checks whether the storage is empty.
        * @return True if size == 0.
        */
        [[nodiscard]]
        bool isEmpty() const;
        /**
         * @brief Checks whether the storage contains valid data.
         * @return True if data pointer is not null.
         */
        [[nodiscard]]
        bool isValid() const;
    private:
        f32* m_data;
        size_t m_size;
        std::pmr::polymorphic_allocator<float> m_alloc;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_ENGINE_STORAGE_STOR_HPP