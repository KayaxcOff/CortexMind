//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_STORAGE_STORAGE_HPP
#define CORTEXMIND_FRAMEWORK_STORAGE_STORAGE_HPP

#include <CortexMind/framework/Memory/device.hpp>
#include <CortexMind/framework/Memory/type.hpp>
#include <tlx/concepts.hpp>
#include <cstddef>

namespace cortex::_fw {
    /**
     * @brief Owns the raw memory associated with a tensor.
     *
     * TensorStorage encapsulates the allocation, ownership, and lifetime
     * of contiguous tensor memory independently of tensor metadata such
     * as shape or data type.
     *
     * Storage may reside in host memory or CUDA device memory depending
     * on the associated @ref TensorDevice.
     *
     * The class provides move-only semantics and automatically releases
     * the owned memory when destroyed.
     */
    class TensorStorage {
    public:
        TensorStorage();
        /**
         * @brief Allocates an empty storage buffer.
         *
         * Creates a storage object with the specified capacity on the
         * requested execution device.
         *
         * @param bytes Number of bytes to allocate.
         * @param type Target device.
         */
        explicit TensorStorage(std::size_t bytes, DeviceType type);
        /**
         * @brief Creates storage initialized from an existing memory buffer.
         *
         * Allocates storage on the specified device and copies the supplied
         * data into the newly allocated memory.
         *
         * @param bytes Number of bytes to copy.
         * @param data Source memory.
         * @param type Destination device.
         *
         * @note The source buffer is expected to reside on the same device
         * specified by @p type.
         */
        TensorStorage(std::size_t bytes, const std::byte* data, DeviceType type);
        TensorStorage(const TensorStorage&) = delete;
        TensorStorage(TensorStorage&& other) noexcept;
        ~TensorStorage();

        template<tlx::arithmetic_like T>
        [[nodiscard]]
        T* as() {
            return reinterpret_cast<T*>(this->m_data);
        }
        template<tlx::arithmetic_like T>
        [[nodiscard]]
        const T* as() const {
            return reinterpret_cast<const T*>(this->m_data);
        }
        /**
         * @brief Returns the underlying memory buffer.
         *
         * @return Pointer to the owned memory.
         */
        [[nodiscard]]
        std::byte* raw() noexcept;
        /**
         * @brief Returns the underlying memory buffer.
         *
         * @return Pointer to the owned memory.
         */
        [[nodiscard]]
        const std::byte* raw() const noexcept;
        /**
         * @brief Returns the storage size in bytes.
         */
        [[nodiscard]]
        std::size_t bytes() const noexcept;
        /**
         * @brief Returns whether the storage contains no bytes.
         */
        [[nodiscard]]
        bool isEmpty() const noexcept;
        /**
         * @brief Returns whether the storage owns a valid memory buffer.
         */
        [[nodiscard]]
        bool isValid() const noexcept;
        [[nodiscard]]
        DeviceType device() const noexcept;
        /**
         * @brief Creates a deep copy of the storage.
         *
         * Allocates a new storage object on the same device and copies all
         * stored bytes.
         *
         * @return A newly allocated storage object containing an identical
         * copy of the underlying memory.
         */
        [[nodiscard]]
        TensorStorage clone() const;

        TensorStorage& operator=(const TensorStorage&) = delete;
        TensorStorage& operator=(TensorStorage&& other) noexcept;
    private:
        std::byte* m_data;
        std::size_t m_bytes;
        TensorDevice m_device;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_STORAGE_STORAGE_HPP