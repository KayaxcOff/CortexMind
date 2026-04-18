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

    struct TensorStorage {
        explicit TensorStorage(size_t size, sys::deviceType device);
        TensorStorage(const TensorStorage& other);
        TensorStorage(TensorStorage&& other) noexcept;
        ~TensorStorage();

        [[nodiscard]]
        f32* data();
        [[nodiscard]]
        const f32 *data() const;
        [[nodiscard]]
        size_t size() const noexcept;
        [[nodiscard]]
        bool isEmpty() const noexcept;
        [[nodiscard]]
        bool isValid() const noexcept;
        [[nodiscard]]
        sys::deviceType device() const noexcept;

        std::vector<i64> shape;
        std::vector<i64> stride;
        i64 offset;

        TensorStorage& operator=(TensorStorage& other);
        TensorStorage& operator=(TensorStorage&& other) noexcept;
    private:
        f32* cpu_ptr;
        f32* gpu_ptr;

        size_t m_size;

        sys::deviceType m_device;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_STORAGE_STOR_HPP