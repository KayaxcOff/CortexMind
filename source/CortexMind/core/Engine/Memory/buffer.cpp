//
// Created by muham on 22.02.2026.
//

#include "CortexMind/core/Engine/Memory/buffer.hpp"
#include <CortexMind/core/Engine/CL2/context.hpp>
#include <CortexMind/core/Tools/err.hpp>

using namespace cortex::_fw::sys;
using namespace cortex::_fw::cl2;

cl_mem_flags buffer::flags() const {
    return this->read_only_ ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
}

buffer::buffer(const size_t count, const bool read_only) : count_(count), read_only_(read_only) {
    CXM_ASSERT(count > 0, "cortex::_fw::buffer::buffer()", "Buffer size can't be zero");

    cl_int err;
    this->buf_ = cl::Buffer(runtime::get().context(), this->flags(), this->bytes(), nullptr, &err);
    CXM_ASSERT(err == CL_SUCCESS, "cortex::_fw::buffer::buffer()", "Buffer couldn't initialized");
}

buffer::buffer(const f32 *src, const size_t count, const bool read_only) : count_(count), read_only_(read_only) {
    CXM_ASSERT(count > 0,  "cortex::_fw::buffer::buffer()", "Buffer size can't be null");
    CXM_ASSERT(src != nullptr, "cortex::_fw::buffer::buffer()", "Source pointer can't be null");

    cl_int err;
    this->buf_ = cl::Buffer(runtime::get().context(), this->flags() | CL_MEM_COPY_HOST_PTR, this->bytes(), const_cast<f32*>(src), &err);
}

buffer::buffer(buffer&& other) noexcept : buf_(std::move(other.buf_)), count_(other.count_), read_only_(other.read_only_){
    other.count_ = 0;
}

buffer& buffer::operator=(buffer&& other) noexcept {
    if (this != &other) {
        this->buf_       = std::move(other.buf_);
        this->count_     = other.count_;
        this->read_only_ = other.read_only_;
        other.count_ = 0;
    }
    return *this;
}

void buffer::upload(const f32* src, size_t count) const {
    if (count == 0) count = this->count_;
    CXM_ASSERT(src != nullptr,  "cortex::_fw::cl2::buffer::upload()", "Source pointer can't be null");
    CXM_ASSERT(count <= this->count_, "cortex::_fw::cl2::buffer::upload()", "Oversize.");

    const cl_int err = runtime::get().queue().enqueueWriteBuffer(
        this->buf_, CL_TRUE, 0, count * sizeof(f32), src
    );
    CXM_ASSERT(err == CL_SUCCESS, "cortex::_fw::cl2::buffer::upload()", "Dataset couldn't upload");
}

void buffer::download(f32* dst, size_t count) const {
    if (count == 0) count = this->count_;
    CXM_ASSERT(dst != nullptr,  "cortex::_fw::cl2::buffer::download()", "Target pointer can't be null.");
    CXM_ASSERT(count <= this->count_, "cortex::_fw::cl2::buffer::download()", "Oversize.");

    const cl_int err = runtime::get().queue().enqueueReadBuffer(
        this->buf_, CL_TRUE, 0, count * sizeof(f32), dst
    );
    CXM_ASSERT(err == CL_SUCCESS, "cortex::_fw::cl2::buffer::download()", "Dataset couldn't download");
}

void buffer::upload(const f32* src, const size_t offset, const size_t count) const {
    CXM_ASSERT(src != nullptr,           "cortex::_fw::cl2::buffer::upload()", "Source pointer can't be null.");
    CXM_ASSERT(offset + count <= this->count_, "cortex::_fw::cl2::buffer::upload()", "Oversize.");

    const cl_int err = runtime::get().queue().enqueueWriteBuffer(
        this->buf_, CL_TRUE, offset * sizeof(f32), count * sizeof(f32), src
    );
    CXM_ASSERT(err == CL_SUCCESS, "cortex::_fw::cl2::buffer::upload()", "Dataset couldn't upload.");
}

void buffer::download(f32* dst, const size_t offset, const size_t count) const {
    CXM_ASSERT(dst != nullptr,           "cortex::_fw::cl2::buffer::download()", "Target pointer can't be null.");
    CXM_ASSERT(offset + count <= this->count_, "cortex::_fw::cl2::buffer::download()", "Oversize.");

    const cl_int err = runtime::get().queue().enqueueReadBuffer(
        this->buf_, CL_TRUE, offset * sizeof(f32), count * sizeof(f32), dst
    );
    CXM_ASSERT(err == CL_SUCCESS, "cortex::_fw::cl2::buffer::download()", "Dataset couldn't download.");
}

const cl::Buffer &buffer::handle() const {
    return this->buf_;
}

size_t buffer::count() const {
    return this->count_;
}

size_t buffer::bytes() const {
    return this->count_ * sizeof(f32);
}
