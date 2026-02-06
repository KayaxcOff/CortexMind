//
// Created by muham on 6.02.2026.
//

#include "CortexMind/core/Engine/Tensor/tensor.hpp"
#include <CortexMind/core/Engine/AVX/matrix.hpp>
#include <CortexMind/core/Engine/AVX/funcs.hpp>
#include <CortexMind/core/Tools/restrict.hpp>
#include <CortexMind/core/Tools/error.hpp>

using namespace cortex::_fw;

MindTensor MindTensor::operator+(const MindTensor &other) const {
    CXM_ASSERT(this->m_shape.size() == other.m_shape.size(), "cortex::_fw::MindTensor::operator+()", "Size must be same");

    MindTensor output(this->m_shape, this->m_grad_flag | other.m_grad_flag);

    const size_t sum = this->numel();

    const f32* restrict dx = this->storage_->data() + this->m_offset;
    const f32* restrict dy = other.storage_->data() + other.m_offset;
    f32* restrict dz = output.storage_->data() + output.m_offset;

    avx2::matrix_t::add(dx, dy, dz, sum);

    return output;
}

MindTensor MindTensor::operator-(const MindTensor &other) const {
    CXM_ASSERT(this->m_shape.size() == other.m_shape.size(), "cortex::_fw::MindTensor::operator-()", "Size must be same");

    MindTensor output(this->m_shape, this->m_grad_flag | other.m_grad_flag);

    const size_t sum = this->numel();

    const f32* restrict dx = this->storage_->data() + this->m_offset;
    const f32* restrict dy = other.storage_->data() + other.m_offset;
    f32* restrict dz = output.storage_->data() + output.m_offset;

    avx2::matrix_t::sub(dx, dy, dz, sum);

    return output;
}

MindTensor MindTensor::operator*(const MindTensor &other) const {
    CXM_ASSERT(this->m_shape.size() == other.m_shape.size(), "cortex::_fw::MindTensor::operator*()", "Size must be same");

    MindTensor output(this->m_shape, this->m_grad_flag | other.m_grad_flag);

    const size_t sum = this->numel();

    const f32* restrict dx = this->storage_->data() + this->m_offset;
    const f32* restrict dy = other.storage_->data() + other.m_offset;
    f32* restrict dz = output.storage_->data() + output.m_offset;

    avx2::matrix_t::mul(dx, dy, dz, sum);

    return output;
}

MindTensor MindTensor::operator/(const MindTensor &other) const {
    CXM_ASSERT(this->m_shape.size() == other.m_shape.size(), "cortex::_fw::MindTensor::operator/()", "Size must be same");

    MindTensor output(this->m_shape, this->m_grad_flag | other.m_grad_flag);

    const size_t sum = this->numel();

    const f32* restrict dx = this->storage_->data() + this->m_offset;
    const f32* restrict dy = other.storage_->data() + other.m_offset;
    f32* restrict dz = output.storage_->data() + output.m_offset;

    avx2::matrix_t::div(dx, dy, dz, sum);

    return output;
}

MindTensor &MindTensor::operator+=(const MindTensor &other) {
    CXM_ASSERT(this->m_shape.size() == other.m_shape.size(), "cortex::_fw::MindTensor::operator+=()", "Size must be same");

    const size_t sum = this->numel();

    f32* dx = this->storage_->data() + this->m_offset;
    const f32* restrict dy = other.storage_->data() + other.m_offset;

    avx2::matrix_t::add(dx, dy, dx, sum);

    return *this;
}

MindTensor &MindTensor::operator-=(const MindTensor &other) {
    CXM_ASSERT(this->m_shape.size() == other.m_shape.size(), "cortex::_fw::MindTensor::operator-=()", "Size must be same");

    const size_t sum = this->numel();

    f32* dx = this->storage_->data() + this->m_offset;
    const f32* restrict dy = other.storage_->data() + other.m_offset;

    avx2::matrix_t::sub(dx, dy, dx, sum);

    return *this;
}

MindTensor &MindTensor::operator*=(const MindTensor &other) {
    CXM_ASSERT(this->m_shape.size() == other.m_shape.size(), "cortex::_fw::MindTensor::operator*=()", "Size must be same");

    const size_t sum = this->numel();

    f32* dx = this->storage_->data() + this->m_offset;
    const f32* restrict dy = other.storage_->data() + other.m_offset;

    avx2::matrix_t::mul(dx, dy, dx, sum);

    return *this;
}

MindTensor &MindTensor::operator/=(const MindTensor &other) {
    CXM_ASSERT(this->m_shape.size() == other.m_shape.size(), "cortex::_fw::MindTensor::operator/=()", "Size must be same");

    const size_t sum = this->numel();

    f32* dx = this->storage_->data() + this->m_offset;
    const f32* restrict dy = other.storage_->data() + other.m_offset;

    avx2::matrix_t::div(dx, dy, dx, sum);

    return *this;
}

MindTensor MindTensor::operator+(const f32 scalar) const {
    MindTensor output(this->m_shape, this->m_grad_flag);

    const size_t sum = this->numel();

    const f32* restrict dx = this->storage_->data() + this->m_offset;
    f32* restrict dy = output.storage_->data() + output.m_offset;
    avx2::vec8f vx = avx2::set(scalar);

    size_t i = 0;
    for (; i + 7 < sum; i += 8) {
        const avx2::vec8f vy = avx2::loadu(dx + i);
        vx = avx2::add(vx, vy);
        avx2::storeu(dy + i, vx);
    }
    for (; i < sum; ++i) dy[i] = dx[i] + scalar;
    return output;
}

MindTensor MindTensor::operator-(const f32 scalar) const {
    MindTensor output(this->m_shape, this->m_grad_flag);

    const size_t sum = this->numel();

    const f32* restrict dx = this->storage_->data() + this->m_offset;
    f32* restrict dy = output.storage_->data() + output.m_offset;
    avx2::vec8f vx = avx2::set(scalar);

    size_t i = 0;
    for (; i + 7 < sum; i += 8) {
        const avx2::vec8f vy = avx2::loadu(dx + i);
        vx = avx2::sub(vx, vy);
        avx2::storeu(dy + i, vx);
    }
    for (; i < sum; ++i) dy[i] = dx[i] - scalar;
    return output;
}

MindTensor MindTensor::operator*(const f32 scalar) const {
    MindTensor output(this->m_shape, this->m_grad_flag);

    const size_t sum = this->numel();

    const f32* restrict dx = this->storage_->data() + this->m_offset;
    f32* restrict dy = output.storage_->data() + output.m_offset;
    avx2::vec8f vx = avx2::set(scalar);

    size_t i = 0;
    for (; i + 7 < sum; i += 8) {
        const avx2::vec8f vy = avx2::loadu(dx + i);
        vx = avx2::mul(vx, vy);
        avx2::storeu(dy + i, vx);
    }
    for (; i < sum; ++i) dy[i] = dx[i] * scalar;
    return output;
}

MindTensor MindTensor::operator/(const f32 scalar) const {
    MindTensor output(this->m_shape, this->m_grad_flag);

    const size_t sum = this->numel();

    const f32* restrict dx = this->storage_->data() + this->m_offset;
    f32* restrict dy = output.storage_->data() + output.m_offset;
    avx2::vec8f vx = avx2::set(scalar);

    size_t i = 0;
    for (; i + 7 < sum; i += 8) {
        const avx2::vec8f vy = avx2::loadu(dx + i);
        vx = avx2::div(vx, vy);
        avx2::storeu(dy + i, vx);
    }
    for (; i < sum; ++i) dy[i] = dx[i] / scalar;
    return output;
}

MindTensor &MindTensor::operator+=(const f32 scalar) {
    const size_t num = this->numel();

    f32* dx = this->storage_->data() + this->m_offset;
    avx2::vec8f vx = avx2::set(scalar);

    size_t i = 0;
    for (; i + 7 < num; i += 8) {
        const avx2::vec8f vy = avx2::loadu(dx + i);
        vx = avx2::add(vx, vy);
        avx2::storeu(dx + i, vx);
    }
    for (; i < num; ++i) dx[i] += scalar;

    return *this;
}

MindTensor &MindTensor::operator-=(const f32 scalar) {
    const size_t num = this->numel();

    f32* dx = this->storage_->data() + this->m_offset;
    avx2::vec8f vx = avx2::set(scalar);

    size_t i = 0;
    for (; i + 7 < num; i += 8) {
        const avx2::vec8f vy = avx2::loadu(dx + i);
        vx = avx2::sub(vx, vy);
        avx2::storeu(dx + i, vx);
    }
    for (; i < num; ++i) dx[i] -= scalar;

    return *this;
}

MindTensor &MindTensor::operator*=(const f32 scalar) {
    const size_t num = this->numel();

    f32* dx = this->storage_->data() + this->m_offset;
    avx2::vec8f vx = avx2::set(scalar);

    size_t i = 0;
    for (; i + 7 < num; i += 8) {
        const avx2::vec8f vy = avx2::loadu(dx + i);
        vx = avx2::mul(vx, vy);
        avx2::storeu(dx + i, vx);
    }
    for (; i < num; ++i) dx[i] *= scalar;

    return *this;
}

MindTensor &MindTensor::operator/=(const f32 scalar) {
    const size_t num = this->numel();

    f32* dx = this->storage_->data() + this->m_offset;
    avx2::vec8f vx = avx2::set(scalar);

    size_t i = 0;
    for (; i + 7 < num; i += 8) {
        const avx2::vec8f vy = avx2::loadu(dx + i);
        vx = avx2::div(vx, vy);
        avx2::storeu(dx + i, vx);
    }
    for (; i < num; ++i) dx[i] /= scalar;

    return *this;
}