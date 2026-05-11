//
// Created by muham on 10.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_STORAGE_STOR_HPP
#define CORTEXMIND_FRAMEWORK_STORAGE_STOR_HPP

#include <CortexMind/framework/Memory/device_type.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Memory/forge.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Memory/mem.hpp>
#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw {
    inline sys::TrackedMem mem;

    #if CXM_IS_CUDA_AVAILABLE
        inline sys::ForgeChunk forge;
    #endif //#if CXM_IS_CUDA_AVAILABLE

    /**
     * @brief Manages the underlying memory storage for tensors.
     *
     * This class handles allocation, device placement (Host vs CUDA),
     * copying, and lifetime management of raw tensor data.
     *
     * @note This is a low-level storage class. Higher-level `Tensor` classes
     *       should build on top of this.
     */
    struct TensorStorage {
        /**
         * @brief Constructs a new storage with specified size and device.
         *
         * @param _size   Number of `f32` elements
         * @param _device Target device (HOST or CUDA)
         */
        TensorStorage(size_t _size, sys::DeviceType _device);
        /**
         * @brief Constructs a new storage with specified size and device.
         *
         * @param _size   Number of `f32` elements
         * @param data    Data
         * @param _device Target device (HOST or CUDA)
         */
        TensorStorage(size_t _size, const f32* data, sys::DeviceType _device);
        TensorStorage(const TensorStorage& other);
        TensorStorage(TensorStorage&& other) noexcept;
        ~TensorStorage();

        /**
         * @brief Returns raw pointer to the data.
         * @return Pointer to host or device memory depending on current device.
         */
        [[nodiscard]]
        f32* data();
        /**
         * @brief Returns raw pointer to the data as const.
         * @return Pointer to host or device memory depending on current device.
         */
        [[nodiscard]]
        const f32* data() const;
        /**
         * @brief Changes the current device and performs necessary data transfer.
         *
         * @param _device New target device
         */
        void SetDevice(sys::DeviceType _device);
        /**
         * @brief Returns size of pointer
         */
        [[nodiscard]]
        size_t size() const;
        /**
         * @brief Returns is size zero
         */
        [[nodiscard]]
        bool isEmpty() const noexcept;
        /**
         * @brief Returns is device pointer null
         */
        [[nodiscard]]
        bool isValid() const noexcept;
        /**
         * @brief Returns the current device type.
         */
        [[nodiscard]]
        sys::DeviceType device() const noexcept;
        /**
         * @brief Creates a deep copy of the storage.
         */
        [[nodiscard]]
        TensorStorage clone() const;

        TensorStorage& operator=(const TensorStorage& other);
        TensorStorage& operator=(TensorStorage&& other) noexcept;
    private:
        f32* m_host_ptr;
        f32* m_cuda_ptr;

        size_t m_size;

        sys::DeviceType m_dev;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_STORAGE_STOR_HPP