//
// Created by muham on 22.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_STORAGE_STOR_HPP
#define CORTEXMIND_CORE_ENGINE_STORAGE_STOR_HPP

#include <CortexMind/core/Engine/Memory/mem.hpp>
#include <CortexMind/core/Engine/Memory/buffer.hpp>
#include <CortexMind/core/Engine/Memory/device.hpp>
#include <CortexMind/core/Tools/params.hpp>
#include <memory>

namespace cortex::_fw {
    struct TensorStorage {
        explicit
        TensorStorage(size_t size, sys::device dev = sys::device::host);
        TensorStorage(const TensorStorage& other);
        TensorStorage(TensorStorage&& other) noexcept;
        ~TensorStorage();

        [[nodiscard]]
        f32* data();
        [[nodiscard]]
        f32* data() const;

        [[nodiscard]]
        sys::buffer* buf();
        [[nodiscard]]
        sys::buffer* buf() const;

        [[nodiscard]]
        TensorStorage to(sys::device target) const;

        [[nodiscard]]
        size_t size() const noexcept;
        [[nodiscard]]
        sys::device kind() const noexcept;
        [[nodiscard]]
        bool is_device(sys::device device) const;
        [[nodiscard]]
        bool isValid() const noexcept;
        [[nodiscard]]
        bool isEmpty() const noexcept;

        TensorStorage& operator=(const TensorStorage& other);
        TensorStorage& operator=(TensorStorage&& other) noexcept;
    private:
        f32* m_data;
        std::unique_ptr<sys::buffer> m_buf;

        size_t m_size;

        sys::device m_device;

        [[nodiscard]]
        static sys::TrackedMem& mem();
        [[nodiscard]]
        static std::pmr::polymorphic_allocator<f32> alloc();

        void allocate_cpu();
        void allocate_gpu();
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_ENGINE_STORAGE_STOR_HPP