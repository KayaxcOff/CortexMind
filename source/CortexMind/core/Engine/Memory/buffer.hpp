//
// Created by muham on 22.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CL2_BUFFER_HPP
#define CORTEXMIND_CORE_ENGINE_CL2_BUFFER_HPP

#include <CortexMind/core/Engine/CL2/params.hpp>
#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::sys {
    class buffer {
    public:
        explicit
        buffer(size_t count, bool read_only = false);
        buffer(const f32* src, size_t count, bool read_only = false);
        buffer(buffer&& other) noexcept;
        buffer& operator=(buffer&& other) noexcept;
        ~buffer() = default;

        buffer(const buffer&)            = delete;
        buffer& operator=(const buffer&) = delete;

        void upload(const f32* src, size_t count = 0) const;
        void download(f32* dst, size_t count = 0) const;

        void upload(const f32* src, size_t offset, size_t count) const;
        void download(f32* dst, size_t offset, size_t count) const;

        [[nodiscard]]
        const cl::Buffer& handle() const;
        [[nodiscard]]
        size_t count() const;
        [[nodiscard]]
        size_t bytes() const;
    private:
        cl::Buffer buf_;
        size_t count_;
        bool read_only_;

        [[nodiscard]]
        cl_mem_flags flags() const;
    };
} // namespace cortex::_fw::sys

#endif //CORTEXMIND_CORE_ENGINE_CL2_BUFFER_HPP