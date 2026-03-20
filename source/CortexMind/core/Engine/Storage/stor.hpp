//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_STORAGE_STOR_HPP
#define CORTEXMIND_CORE_ENGINE_STORAGE_STOR_HPP

#include <CortexMind/core/Engine/Memory/buf.cuh>
#include <CortexMind/core/Engine/Memory/device.hpp>
#include <CortexMind/core/Engine/Memory/mem.hpp>

namespace cortex::_fw {
    // Global memory pools (singletons)
    inline sys::TrackedMem mem;     ///< Global host memory pool
    inline sys::DeviceHeap heap;    ///< Global device memory pool
    /**
     * @brief   RAII owner of a contiguous float buffer on host or device
     *
     * Allocates memory via TrackedMem (CPU) or DeviceHeap (GPU) and manages its lifetime.
     * Supports explicit transfer between host and device (.to(dev)).
     *
     * @note    Only raw data pointer and size — no shape/stride/dtype
     * @note    Copy constructor performs deep copy (expensive)
     * @note    Move constructor transfers ownership (cheap)
     * @note    Destructor automatically deallocates from correct pool
     */
    struct TensorStorage {
        /**
         * @brief   Constructs storage with given size on specified device
         * @param   size    Number of float elements
         * @param   d       Target device (dev::host or dev::cuda)
         *
         * @note    Allocates on host always; on device only if d == dev::cuda
         * @note    Throws via CXM_ASSERT on allocation failure
         */
        explicit
        TensorStorage(size_t size, sys::dev d);
        /**
         * @brief   Copy constructor – deep copy from another storage
         * @param   other   Source TensorStorage
         *
         * @note    Allocates new buffers and copies data
         * @note    Expensive operation — prefer move when possible
         */
        TensorStorage(const TensorStorage& other);
        /**
         * @brief   Move constructor – transfers ownership
         * @param   other   Source (becomes empty after move)
         */
        TensorStorage(TensorStorage&& other) noexcept;
        /**
         * @brief   Destructor – frees memory from correct pool
         */
        ~TensorStorage();

        /**
         * @brief   Returns raw data pointer (host or device depending on current dev)
         * @return  float* pointing to data
         */
        [[nodiscard]]
        f32* data();
        /**
         * @brief   Const version of data()
         */
        [[nodiscard]]
        f32* data() const;
        /**
         * @brief   Returns number of float elements
         */
        [[nodiscard]]
        size_t size() const noexcept;
        /**
         * @brief   Checks if storage is empty (size == 0)
         */
        [[nodiscard]]
        bool isEmpty() const noexcept;
        /**
         * @brief   Checks if storage has valid allocation
         */
        [[nodiscard]]
        bool isValid() const;
        /**
         * @brief   Returns true if currently on host
         */
        [[nodiscard]]
        bool is_cpu() const;
        /**
         * @brief   Returns true if currently on device (GPU)
         */
        [[nodiscard]]
        bool is_gpu() const;

        /**
         * @brief   Transfers data to target device (if not already there)
         * @param   d       Target device (dev::host or dev::cuda)
         *
         * @note    Allocates missing buffer if needed
         * @note    Copies data synchronously (upload/download)
         * @note    No-op if already on target device
         */
        void to(sys::dev d);
    private:
        f32* m_host;
        f32* m_device;

        size_t m_size;
        sys::dev m_dev;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_ENGINE_STORAGE_STOR_HPP