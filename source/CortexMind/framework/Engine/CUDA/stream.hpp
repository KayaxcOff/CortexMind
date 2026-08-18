//
// Created by muham on 18.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_STREAM_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_STREAM_HPP

#include <driver_types.h>

namespace cortex::_fw::nv {
    struct stream {
        stream();
        stream(const stream&) = delete;
        stream(stream&& other) noexcept;
        ~stream();

        [[nodiscard]]
        operator cudaStream_t() const noexcept;

        void synchronize() const noexcept;

        stream& operator=(const stream&) = delete;
        stream& operator=(stream&& other) noexcept;
    private:
        cudaStream_t m_value;
    };
} //namespace cortex::_fw::nv

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_STREAM_HPP