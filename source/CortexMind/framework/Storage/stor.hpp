//
// Created by muham on 18.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_STORAGE_STOR_HPP
#define CORTEXMIND_FRAMEWORK_STORAGE_STOR_HPP

#include <CortexMind/framework/Tools/params.hpp>
#include <CortexMind/framework/Memory/device.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Memory/forge.hpp>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Memory/mem.hpp>
#include <vector>

namespace cortex::_fw {
    inline sys::TrackedMem mem;
    #if CXM_IS_CUDA_AVAILABLE
        inline sys::ForgeChunk forge;
    #endif //#if CXM_IS_CUDA_AVAILABLE

    /**
     * @brief Core storage class for Tensor data and metadata.
     *
     * Holds the actual data pointer (CPU or GPU), size, shape, stride and offset.
     * Designed to support automatic differentiation by keeping metadata alive
     * even if the original Tensor object goes out of scope.
     */
    struct TensorStorage {
        /**
         * @brief Constructs a TensorStorage with given size and device.
         * @param size   Number of elements
         * @param device Host or CUDA device
         */
        explicit TensorStorage(size_t size, sys::deviceType device);
        TensorStorage(const TensorStorage& other);
        TensorStorage(TensorStorage&& other) noexcept;
        ~TensorStorage();

        /**
         * @brief Returns mutable pointer to the underlying data.
         * @return `cpu_ptr` if on host, `gpu_ptr` if on CUDA
         */
        [[nodiscard]]
        f32* data();
        /**
         * @brief Returns const pointer to the underlying data.
         * @return `cpu_ptr` if on host, `gpu_ptr` if on CUDA
         */
        [[nodiscard]]
        const f32 *data() const;
        /**
         * @brief Returns the number of elements in the storage.
         */
        [[nodiscard]]
        size_t size() const noexcept;
        /**
         * @brief Checks if the storage has zero elements.
         */
        [[nodiscard]]
        bool isEmpty() const noexcept;
        /**
         * @brief Checks if the storage has a valid data pointer.
         */
        [[nodiscard]]
        bool isValid() const noexcept;
        /**
         * @brief Returns the current device type (host or cuda).
         */
        [[nodiscard]]
        sys::deviceType device() const noexcept;
        /**
         * @brief Changes the device type (does not move data).
         * @param device New device type
         */
        void setDevice(sys::deviceType device) noexcept;

        // Why is metadata inside TensorStorage? Because it's for automatic
        // gradient calculation. We'll use the tensor class as an object. This
        // means that if the tensor variable is never used again, it will die,
        // but we need its data and metadata for automatic gradient calculation.
        // TensorStorage will be stored inside the tensor as `shared_ptr<>`, meaning
        // that even if the tensor object dies, TensorStorage will survive. Since
        // TensorStorage holds the data, if we send it to the automatic gradient
        // calculation, we can find the derivative of the changing values, but not
        // only the data, but also the tensor's metadata is needed for the operations.
        // That's why metadata is inside TensorStorage.

        std::vector<i64> shape;
        std::vector<i64> stride;
        i64 offset;

        TensorStorage& operator=(const TensorStorage& other);
        TensorStorage& operator=(TensorStorage&& other) noexcept;
    private:
        f32* cpu_ptr;
        f32* gpu_ptr;

        size_t m_size;

        sys::deviceType m_device;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_STORAGE_STOR_HPP