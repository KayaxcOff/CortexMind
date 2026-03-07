//
// Created by muham on 22.02.2026.
//

#include "CortexMind/core/Engine/Tensor/tensor.hpp"
#include <CortexMind/core/Engine/AVX2/matrix.hpp>
#include <CortexMind/core/Engine/AVX2/funcs.hpp>
#include <CortexMind/core/Engine/CL2/funcs.hpp>
#include <CortexMind/core/Graph/flow_ops.hpp>
#include <CortexMind/core/Tools/restrict.hpp>
#include <CortexMind/core/Tools/err.hpp>

using namespace cortex::_fw;
using namespace cortex::_fw::sys;

MindTensor MindTensor::operator+(const MindTensor &other) const {
    CXM_ASSERT(this->m_device == other.m_device, "cortex::_fw::MindTensor::operator+()", "Devices mismatch");

    const std::vector<i64> out_shape = broadcast_shape(this->m_shape, other.m_shape);

    MindTensor a = (this->m_shape == out_shape) ? *this : this->expand(out_shape).clone();
    MindTensor b = (other.m_shape == out_shape) ? other : other.expand(out_shape).clone();

    MindTensor output(out_shape, this->m_device, this->m_require || other.m_require);

    if (output.m_device == device::host) {
        avx2::matrix_t::add(a.get(), b.get(), output.get(), output.numel());
    } else if (output.m_device == device::cuda) {
        cl2::add(*a.buffer(), *b.buffer(), *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator+()", "Invalid device");
    }

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::add>(
            const_cast<MindTensor*>(this),
            const_cast<MindTensor*>(&other)
        );
    }
    return output;
}

MindTensor MindTensor::operator-(const MindTensor &other) const {
    CXM_ASSERT(this->m_device == other.m_device, "cortex::_fw::MindTensor::operator-()", "Devices mismatch");

    const std::vector<i64> out_shape = broadcast_shape(this->m_shape, other.m_shape);

    MindTensor a = (this->m_shape == out_shape) ? *this : this->expand(out_shape).clone();
    MindTensor b = (other.m_shape == out_shape) ? other : other.expand(out_shape).clone();

    MindTensor output(out_shape, this->m_device, this->m_require || other.m_require);

    if (output.m_device == device::host) {
        avx2::matrix_t::sub(a.get(), b.get(), output.get(), output.numel());
    } else if (output.m_device == device::cuda) {
        cl2::sub(*a.buffer(), *b.buffer(), *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator-()", "Invalid device");
    }

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::sub>(
            const_cast<MindTensor*>(this),
            const_cast<MindTensor*>(&other));
    }
    return output;
}

MindTensor MindTensor::operator*(const MindTensor &other) const {
    CXM_ASSERT(this->m_device == other.m_device, "cortex::_fw::MindTensor::operator*()", "Devices mismatch");

    const std::vector<i64> out_shape = broadcast_shape(this->m_shape, other.m_shape);

    MindTensor a = (this->m_shape == out_shape) ? *this : this->expand(out_shape).clone();
    MindTensor b = (other.m_shape == out_shape) ? other : other.expand(out_shape).clone();

    MindTensor output(out_shape, this->m_device, this->m_require || other.m_require);

    if (output.m_device == device::host) {
        avx2::matrix_t::mul(a.get(), b.get(), output.get(), output.numel());
    } else if (output.m_device == device::cuda) {
        cl2::mul(*a.buffer(), *b.buffer(), *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator*()", "Invalid device");
    }

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::mul>(
            const_cast<MindTensor*>(this),
            const_cast<MindTensor*>(&other)
        );
    }
    return output;
}

MindTensor MindTensor::operator/(const MindTensor &other) const {
    CXM_ASSERT(this->m_device == other.m_device, "cortex::_fw::MindTensor::operator/()", "Devices mismatch");

    const std::vector<i64> out_shape = broadcast_shape(this->m_shape, other.m_shape);

    MindTensor a = (this->m_shape == out_shape) ? *this : this->expand(out_shape).clone();
    MindTensor b = (other.m_shape == out_shape) ? other : other.expand(out_shape).clone();

    MindTensor output(out_shape, this->m_device, this->m_require || other.m_require);

    if (output.m_device == device::host) {
        avx2::matrix_t::div(a.get(), b.get(), output.get(), output.numel());
    } else if (output.m_device == device::cuda) {
        cl2::div(*a.buffer(), *b.buffer(), *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator/()", "Invalid device");
    }

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::div>(
            const_cast<MindTensor*>(this),
            const_cast<MindTensor*>(&other)
        );
    }
    return output;
}

MindTensor& MindTensor::operator+=(const MindTensor& other) {
    CXM_ASSERT(this->m_device == other.m_device, "cortex::_fw::MindTensor::operator+=()", "Devices mismatch");
    MindTensor b = (other.m_shape == this->m_shape)
                   ? other : other.expand(this->m_shape).clone();
    if (this->m_device == device::host) {
        avx2::matrix_t::add(this->get(), b.get(), this->get(), this->numel());
    } else if (this->m_device == device::cuda) {
        cl2::add(*this->buffer(), *b.buffer(), *this->buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator+=()", "Invalid device");
    }
    return *this;
}

MindTensor& MindTensor::operator-=(const MindTensor& other) {
    CXM_ASSERT(this->m_device == other.m_device, "cortex::_fw::MindTensor::operator-=()", "Devices mismatch");
    MindTensor b = (other.m_shape == this->m_shape)
                   ? other : other.expand(this->m_shape).clone();
    if (this->m_device == device::host) {
        avx2::matrix_t::sub(this->get(), b.get(), this->get(), this->numel());
    } else if (this->m_device == device::cuda) {
        cl2::sub(*this->buffer(), *b.buffer(), *this->buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator-=()", "Invalid device");
    }
    return *this;
}

MindTensor& MindTensor::operator*=(const MindTensor& other) {
    CXM_ASSERT(this->m_device == other.m_device, "cortex::_fw::MindTensor::operator*=()", "Devices mismatch");
    MindTensor b = (other.m_shape == this->m_shape)
                   ? other : other.expand(this->m_shape).clone();
    if (this->m_device == device::host) {
        avx2::matrix_t::mul(this->get(), b.get(), this->get(), this->numel());
    } else if (this->m_device == device::cuda) {
        cl2::mul(*this->buffer(), *b.buffer(), *this->buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator*=()", "Invalid device");
    }
    return *this;
}

MindTensor& MindTensor::operator/=(const MindTensor& other) {
    CXM_ASSERT(this->m_device == other.m_device, "cortex::_fw::MindTensor::operator/=()", "Devices mismatch");
    MindTensor b = (other.m_shape == this->m_shape)
                   ? other : other.expand(this->m_shape).clone();
    if (this->m_device == device::host) {
        avx2::matrix_t::div(this->get(), b.get(), this->get(), this->numel());
    } else if (this->m_device == device::cuda) {
        cl2::div(*this->buffer(), *b.buffer(), *this->buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator/=()", "Invalid device");
    }
    return *this;
}

MindTensor MindTensor::operator+(const f32 scalar) const {
    MindTensor output(this->m_shape, this->m_device, this->m_require);

    if (this->m_device == device::host) {
        const f32* restrict px = this->get();
        f32*       restrict py = output.get();
        const size_t num = this->numel();

        const avx2::vec8f vs = avx2::set1(scalar);
        size_t i = 0;
        for (; i + 8 <= num; i += 8)
            avx2::storeu(py + i, avx2::add(avx2::loadu(px + i), vs));
        for (; i < num; ++i) py[i] = px[i] + scalar;

    } else if (this->m_device == device::cuda) {
        cl2::add(*this->buffer(), scalar, *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator+(f32)", "Invalid device.");
    }
    return output;
}

MindTensor MindTensor::operator-(const f32 scalar) const {
    MindTensor output(this->m_shape, this->m_device, this->m_require);

    if (this->m_device == device::host) {
        const f32* restrict px = this->get();
        f32*       restrict py = output.get();
        const size_t num = this->numel();

        const avx2::vec8f vs = avx2::set1(scalar);
        size_t i = 0;
        for (; i + 8 <= num; i += 8)
            avx2::storeu(py + i, avx2::sub(avx2::loadu(px + i), vs));
        for (; i < num; ++i) py[i] = px[i] - scalar;

    } else if (this->m_device == device::cuda) {
        cl2::add(*this->buffer(), -scalar, *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator-(f32)", "Invalid device.");
    }
    return output;
}

MindTensor MindTensor::operator*(const f32 scalar) const {
    MindTensor output(this->m_shape, this->m_device, this->m_require);

    if (this->m_device == device::host) {
        const f32* restrict px = this->get();
        f32*       restrict py = output.get();
        const size_t num = this->numel();

        const avx2::vec8f vs = avx2::set1(scalar);
        size_t i = 0;
        for (; i + 8 <= num; i += 8)
            avx2::storeu(py + i, avx2::mul(avx2::loadu(px + i), vs));
        for (; i < num; ++i) py[i] = px[i] * scalar;

    } else if (this->m_device == device::cuda) {
        cl2::mul(*this->buffer(), scalar, *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator*(f32)", "Invalid device.");
    }
    return output;
}

MindTensor MindTensor::operator/(const f32 scalar) const {
    CXM_ASSERT(scalar != 0.0f,
               "cortex::_fw::MindTensor::operator/(f32)", "Division by zero.");

    MindTensor output(this->m_shape, this->m_device, this->m_require);

    if (this->m_device == device::host) {
        const f32* restrict px = this->get();
        f32*       restrict py = output.get();
        const size_t num = this->numel();

        const f32 inv = 1.0f / scalar;
        const avx2::vec8f vs = avx2::set1(inv);
        size_t i = 0;
        for (; i + 8 <= num; i += 8)
            avx2::storeu(py + i, avx2::mul(avx2::loadu(px + i), vs));
        for (; i < num; ++i) py[i] = px[i] * inv;

    } else if (this->m_device == device::cuda) {
        cl2::mul(*this->buffer(), 1.0f / scalar, *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator/(f32)", "Invalid device.");
    }
    return output;
}

MindTensor& MindTensor::operator+=(const f32 scalar) {
    CXM_WARN_IF(!this->m_require,
                "cortex::_fw::MindTensor::operator+=(f32)",
                "In-place op on tensor that requires grad may corrupt autograd graph.");

    if (this->m_device == device::host) {
        f32* restrict px = this->get();
        const size_t num = this->numel();
        const avx2::vec8f vs = avx2::set1(scalar);
        size_t i = 0;
        for (; i + 8 <= num; i += 8)
            avx2::storeu(px + i, avx2::add(avx2::loadu(px + i), vs));
        for (; i < num; ++i) px[i] += scalar;

    } else if (this->m_device == device::cuda) {
        cl2::add(*this->buffer(), scalar, *this->buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator+=(f32)", "Invalid device.");
    }
    return *this;
}

MindTensor& MindTensor::operator-=(const f32 scalar) {
    CXM_WARN_IF(!this->m_require,
                "cortex::_fw::MindTensor::operator-=(f32)",
                "In-place op on tensor that requires grad may corrupt autograd graph.");

    if (this->m_device == device::host) {
        f32* restrict px = this->get();
        const size_t num = this->numel();
        const avx2::vec8f vs = avx2::set1(scalar);
        size_t i = 0;
        for (; i + 8 <= num; i += 8)
            avx2::storeu(px + i, avx2::sub(avx2::loadu(px + i), vs));
        for (; i < num; ++i) px[i] -= scalar;

    } else if (this->m_device == device::cuda) {
        cl2::add(*this->buffer(), -scalar, *this->buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator-=(f32)", "Invalid device.");
    }
    return *this;
}

MindTensor& MindTensor::operator*=(const f32 scalar) {
    CXM_WARN_IF(!this->m_require,
                "cortex::_fw::MindTensor::operator*=(f32)",
                "In-place op on tensor that requires grad may corrupt autograd graph.");

    if (this->m_device == device::host) {
        f32* restrict px = this->get();
        const size_t num = this->numel();
        const avx2::vec8f vs = avx2::set1(scalar);
        size_t i = 0;
        for (; i + 8 <= num; i += 8)
            avx2::storeu(px + i, avx2::mul(avx2::loadu(px + i), vs));
        for (; i < num; ++i) px[i] *= scalar;

    } else if (this->m_device == device::cuda) {
        cl2::mul(*this->buffer(), scalar, *this->buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator*=(f32)", "Invalid device.");
    }
    return *this;
}

MindTensor& MindTensor::operator/=(const f32 scalar) {
    CXM_ASSERT(scalar != 0.0f,
               "cortex::_fw::MindTensor::operator/=(f32)", "Division by zero.");
    CXM_WARN_IF(!this->m_require,
                "cortex::_fw::MindTensor::operator/=(f32)",
                "In-place op on tensor that requires grad may corrupt autograd graph.");

    const f32 inv = 1.0f / scalar;

    if (this->m_device == device::host) {
        f32* restrict px = this->get();
        const size_t num = this->numel();
        const avx2::vec8f vs = avx2::set1(inv);
        size_t i = 0;
        for (; i + 8 <= num; i += 8)
            avx2::storeu(px + i, avx2::mul(avx2::loadu(px + i), vs));
        for (; i < num; ++i) px[i] *= inv;

    } else if (this->m_device == device::cuda) {
        cl2::mul(*this->buffer(), inv, *this->buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator/=(f32)", "Invalid device.");
    }
    return *this;
}

MindTensor cortex::_fw::operator+(const f32 scalar, const MindTensor& t) {
    return t + scalar;
}

MindTensor cortex::_fw::operator*(const f32 scalar, const MindTensor& t) {
    return t * scalar;
}

bool MindTensor::operator==(const MindTensor& other) const {
    if (this->m_shape != other.m_shape)   return false;
    if (this->m_device != other.m_device) return false;

    const size_t num = this->numel();

    if (this->m_device == device::host) {
        const f32* restrict px = this->get();
        const f32* restrict py = other.get();
        for (size_t i = 0; i < num; ++i)
            if (px[i] != py[i]) return false;
        return true;
    }

    std::vector<f32> a(num), b(num);
    this->buffer()->download(a.data(), num);
    other.buffer()->download(b.data(), num);
    return a == b;
}

bool MindTensor::operator!=(const MindTensor& other) const {
    return !(*this == other);
}

MindTensor &MindTensor::operator=(const MindTensor &other) {
    if (this == &other) return *this;

    this->m_shape   = other.m_shape;
    this->m_stride  = other.m_stride;
    this->m_offset  = other.m_offset;
    this->m_device  = other.m_device;
    this->m_require = other.m_require;

    this->storage_ = std::make_shared<TensorStorage>(*other.storage_);
    this->flow_    = other.flow_;

    this->gradient_ = other.gradient_ ? std::make_unique<MindTensor>(*other.gradient_) : nullptr;

    return *this;
}

MindTensor &MindTensor::operator=(MindTensor &&other) noexcept {
    if (this == &other) return *this;

    this->m_shape   = std::move(other.m_shape);
    this->m_stride  = std::move(other.m_stride);
    this->m_offset  = other.m_offset;
    this->m_device  = other.m_device;
    this->m_require = other.m_require;

    this->storage_  = std::move(other.storage_);
    this->flow_     = std::move(other.flow_);
    this->gradient_ = std::move(other.gradient_);

    other.m_offset  = 0;
    other.m_require = false;

    return *this;
}