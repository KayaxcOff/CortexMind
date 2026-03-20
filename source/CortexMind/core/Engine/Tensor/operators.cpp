//
// Created by muham on 16.03.2026.
//

#include "CortexMind/core/Engine/Tensor/tensor.hpp"
#include <CortexMind/core/Engine/AVX2/inplace.hpp>
#include <CortexMind/core/Engine/AVX2/matrix.hpp>
#include <CortexMind/core/Engine/AVX2/scalar.hpp>
#include <CortexMind/core/Engine/CUDA/broadcast.cuh>
#include <CortexMind/core/Engine/CUDA/inplace.cuh>
#include <CortexMind/core/Engine/CUDA/matrix.cuh>
#include <CortexMind/core/Engine/CUDA/scalar.cuh>
#include <CortexMind/core/Graph/ops.hpp>
#include <type_traits>

using namespace cortex::_fw;
using namespace cortex::_fw::sys;

template<typename CpuOp, typename CudaFn>
static MindTensor broadcast_op(
    const MindTensor& lhs, const MindTensor& rhs,
    CpuOp cpu_op, CudaFn cuda_fn,
    const std::vector<i64>& out_shape)
{
    const size_t ndim = out_shape.size();
    const std::vector<i64> sx  = pad_shape(lhs.shape(), ndim);
    const std::vector<i64> sy  = pad_shape(rhs.shape(), ndim);
    const std::vector<i64> stx = pad_stride(lhs.stride(), lhs.shape(), ndim);
    const std::vector<i64> sty = pad_stride(rhs.stride(), rhs.shape(), ndim);

    MindTensor output(out_shape, lhs.device());

    if (lhs.device() == dev::host) {
        broadcast_cpu(lhs.get(), rhs.get(), output.get(), sx, stx, sy, sty, out_shape, cpu_op);
    } else if (lhs.device() == dev::cuda) {
        cuda_fn(lhs.get(), rhs.get(), output.get(), sx, stx, sy, sty, out_shape);
    } else {
        CXM_ASSERT(false, "broadcast_op()", "Invalid device");
    }
    return output;
}

MindTensor MindTensor::operator+(const MindTensor &other) {
    CXM_ASSERT(this->m_dev == other.m_dev,
        "cortex::_fw::MindTensor::operator+()", "Tensors must be on same device");
    CXM_ASSERT(is_broadcastable(this->m_shape, other.m_shape),
        "cortex::_fw::MindTensor::operator+()", "Shapes are not broadcastable");

    const std::vector<i64> out_shape = broadcast_shape(this->m_shape, other.m_shape);
    MindTensor output;

    if (this->m_shape == other.m_shape) {
        output = MindTensor(out_shape, this->m_dev, this->m_grad_flag || other.m_grad_flag);
        if (this->m_dev == dev::host)
            avx2::matrix_t::add(this->get(), other.get(), output.get(), this->numel());
        else if (this->m_dev == dev::cuda)
            cuda::matrix_t::add(this->get(), other.get(), output.get(), this->numel());
        else CXM_ASSERT(false, "cortex::_fw::MindTensor::operator+()", "Invalid device");
    } else {
        output = broadcast_op(*this, other,
            [](const f32 a, const f32 b) { return a + b; },
            [](const f32* x, const f32* y, f32* z,
               const std::vector<i64>& sx, const std::vector<i64>& stx,
               const std::vector<i64>& sy, const std::vector<i64>& sty,
               const std::vector<i64>& out) {
                cuda::broadcast_t::add(x, y, z, sx, stx, sy, sty, out);
            }, out_shape);
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::add>(this, const_cast<MindTensor*>(&other));
    }

    return output;
}

MindTensor MindTensor::operator-(const MindTensor &other) {
    CXM_ASSERT(this->m_dev == other.m_dev,
        "cortex::_fw::MindTensor::operator-()", "Tensors must be on same device");
    CXM_ASSERT(is_broadcastable(this->m_shape, other.m_shape),
        "cortex::_fw::MindTensor::operator-()", "Shapes are not broadcastable");

    const std::vector<i64> out_shape = broadcast_shape(this->m_shape, other.m_shape);
    MindTensor output;

    if (this->m_shape == other.m_shape) {
        output = MindTensor(out_shape, this->m_dev, this->m_grad_flag || other.m_grad_flag);
        if (this->m_dev == dev::host)
            avx2::matrix_t::sub(this->get(), other.get(), output.get(), this->numel());
        else if (this->m_dev == dev::cuda)
            cuda::matrix_t::sub(this->get(), other.get(), output.get(), this->numel());
        else CXM_ASSERT(false, "cortex::_fw::MindTensor::operator-()", "Invalid device");
    } else {
        output = broadcast_op(*this, other,
            [](const f32 a, const f32 b) { return a - b; },
            [](const f32* x, const f32* y, f32* z,
               const std::vector<i64>& sx, const std::vector<i64>& stx,
               const std::vector<i64>& sy, const std::vector<i64>& sty,
               const std::vector<i64>& out) {
                cuda::broadcast_t::sub(x, y, z, sx, stx, sy, sty, out);
            }, out_shape);
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::sub>(this, const_cast<MindTensor*>(&other));
    }

    return output;
}

MindTensor MindTensor::operator*(const MindTensor &other) {
    CXM_ASSERT(this->m_dev == other.m_dev,
        "cortex::_fw::MindTensor::operator*()", "Tensors must be on same device");
    CXM_ASSERT(is_broadcastable(this->m_shape, other.m_shape),
        "cortex::_fw::MindTensor::operator*()", "Shapes are not broadcastable");

    const std::vector<i64> out_shape = broadcast_shape(this->m_shape, other.m_shape);
    MindTensor output;

    if (this->m_shape == other.m_shape) {
        output = MindTensor(out_shape, this->m_dev, this->m_grad_flag || other.m_grad_flag);
        if (this->m_dev == dev::host)
            avx2::matrix_t::mul(this->get(), other.get(), output.get(), this->numel());
        else if (this->m_dev == dev::cuda)
            cuda::matrix_t::mul(this->get(), other.get(), output.get(), this->numel());
        else CXM_ASSERT(false, "cortex::_fw::MindTensor::operator*()", "Invalid device");
    } else {
        output = broadcast_op(*this, other,
            [](const f32 a, const f32 b) { return a * b; },
            [](const f32* x, const f32* y, f32* z,
               const std::vector<i64>& sx, const std::vector<i64>& stx,
               const std::vector<i64>& sy, const std::vector<i64>& sty,
               const std::vector<i64>& out) {
                cuda::broadcast_t::mul(x, y, z, sx, stx, sy, sty, out);
            }, out_shape);
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::div>(this, const_cast<MindTensor*>(&other));
    }

    return output;
}

MindTensor MindTensor::operator/(const MindTensor &other) {
    CXM_ASSERT(this->m_dev == other.m_dev,
        "cortex::_fw::MindTensor::operator/()", "Tensors must be on same device");
    CXM_ASSERT(is_broadcastable(this->m_shape, other.m_shape),
        "cortex::_fw::MindTensor::operator/()", "Shapes are not broadcastable");

    const std::vector<i64> out_shape = broadcast_shape(this->m_shape, other.m_shape);
    MindTensor output;

    if (this->m_shape == other.m_shape) {
        output = MindTensor(out_shape, this->m_dev, this->m_grad_flag || other.m_grad_flag);
        if (this->m_dev == dev::host)
            avx2::matrix_t::div(this->get(), other.get(), output.get(), this->numel());
        else if (this->m_dev == dev::cuda)
            cuda::matrix_t::div(this->get(), other.get(), output.get(), this->numel());
        else CXM_ASSERT(false, "cortex::_fw::MindTensor::operator/()", "Invalid device");
    } else {
        output = broadcast_op(*this, other,
            [](const f32 a, const f32 b) { return a / b; },
            [](const f32* x, const f32* y, f32* z,
               const std::vector<i64>& sx, const std::vector<i64>& stx,
               const std::vector<i64>& sy, const std::vector<i64>& sty,
               const std::vector<i64>& out) {
                cuda::broadcast_t::div(x, y, z, sx, stx, sy, sty, out);
            }, out_shape);
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::div>(this, const_cast<MindTensor*>(&other));
    }

    return output;
}

MindTensor &MindTensor::operator+=(const MindTensor &other) {
    CXM_ASSERT(this->m_dev == other.m_dev,
        "cortex::_fw::MindTensor::operator+=()", "Tensors must be on same device");
    CXM_ASSERT(this->m_shape == other.m_shape,
        "cortex::_fw::MindTensor::operator+=()", "Shapes must match for in-place operation");

    if (this->m_dev == dev::host) {
        avx2::inplace::add(this->get(), other.get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::inplace::add(this->get(), other.get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator+=()", "Invalid device");
    }

    return *this;
}

MindTensor &MindTensor::operator-=(const MindTensor &other) {
    CXM_ASSERT(this->m_dev == other.m_dev,
        "cortex::_fw::MindTensor::operator-=()", "Tensors must be on same device");
    CXM_ASSERT(this->m_shape == other.m_shape,
        "cortex::_fw::MindTensor::operator-=()", "Shapes must match for in-place operation");

    if (this->m_dev == dev::host) {
        avx2::inplace::sub(this->get(), other.get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::inplace::sub(this->get(), other.get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator-=()", "Invalid device");
    }

    return *this;
}

MindTensor &MindTensor::operator*=(const MindTensor &other) {
    CXM_ASSERT(this->m_dev == other.m_dev,
        "cortex::_fw::MindTensor::operator*=()", "Tensors must be on same device");
    CXM_ASSERT(this->m_shape == other.m_shape,
        "cortex::_fw::MindTensor::operator*=()", "Shapes must match for in-place operation");

    if (this->m_dev == dev::host) {
        avx2::inplace::mul(this->get(), other.get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::inplace::mul(this->get(), other.get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator*=()", "Invalid device");
    }

    return *this;
}

MindTensor &MindTensor::operator/=(const MindTensor &other) {
    CXM_ASSERT(this->m_dev == other.m_dev,
        "cortex::_fw::MindTensor::operator/=()", "Tensors must be on same device");
    CXM_ASSERT(this->m_shape == other.m_shape,
        "cortex::_fw::MindTensor::operator/=()", "Shapes must match for in-place operation");

    if (this->m_dev == dev::host) {
        avx2::inplace::div(this->get(), other.get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::inplace::div(this->get(), other.get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator/=()", "Invalid device");
    }

    return *this;
}

MindTensor MindTensor::operator+(const f32 value) {
    MindTensor output(this->shape(), this->device());

    if (this->m_dev == dev::host) {
        avx2::ScalarOp::add(this->get(), value, output.get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::ScalarKernel::add(this->get(), value, output.get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator+()", "Invalid device");
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::add_scalar>(this, value);
    }

    return output;
}

MindTensor MindTensor::operator-(const f32 value) {
    MindTensor output(this->shape(), this->device());

    if (this->m_dev == dev::host) {
        avx2::ScalarOp::sub(this->get(), value, output.get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::ScalarKernel::sub(this->get(), value, output.get(), this->numel());
    }  else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator-()", "Invalid device");
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::sub_scalar>(this, value);
    }

    return output;
}

MindTensor MindTensor::operator*(const f32 value) {
    MindTensor output(this->shape(), this->device());

    if (this->m_dev == dev::host) {
        avx2::ScalarOp::mul(this->get(), value, output.get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::ScalarKernel::mul(this->get(), value, output.get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator*()", "Invalid device");
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::div_scalar>(this, value);
    }

    return output;
}

MindTensor MindTensor::operator/(const f32 value) {
    MindTensor output(this->shape(), this->device());

    if (this->m_dev == dev::host) {
        avx2::ScalarOp::div(this->get(), value, output.get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::ScalarKernel::div(this->get(), value, output.get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator/()", "Invalid device");
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::div_scalar>(this, value);
    }

    return output;
}

MindTensor &MindTensor::operator+=(const f32 value) {
    if (this->m_dev == dev::host) {
        avx2::inplace::add(this->get(), value, this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::inplace_scalar::add(this->get(), value, this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator+=()", "Invalid device");
    }

    return *this;
}

MindTensor &MindTensor::operator-=(const f32 value) {
    if (this->m_dev == dev::host) {
        avx2::inplace::sub(this->get(), value, this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::inplace_scalar::sub(this->get(), value, this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator-=()", "Invalid device");
    }

    return *this;
}

MindTensor &MindTensor::operator*=(const f32 value) {
    if (this->m_dev == dev::host) {
        avx2::inplace::mul(this->get(), value, this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::inplace_scalar::mul(this->get(), value, this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator*=()", "Invalid device");
    }

    return *this;
}

MindTensor &MindTensor::operator/=(const f32 value) {
    if (this->m_dev == dev::host) {
        avx2::inplace::div(this->get(), value, this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::inplace_scalar::div(this->get(), value, this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::operator/=()", "Invalid device");
    }

    return *this;
}

bool MindTensor::operator==(const MindTensor &other) const {
    if (this->m_shape != other.m_shape) return false;
    if (this->m_dev != other.m_dev) return false;

    bool output = false;

    if (this->m_dev == dev::host) {
        const f32* __restrict px = this->get();
        const f32* __restrict py = other.get();
        for (size_t i = 0; i < this->numel(); ++i) {
            if (px[i] != py[i]) return false;
        }
        output = true;
    }
    return output;
}

namespace cortex::_fw {
    MindTensor operator+(const f32 scalar, MindTensor& t) {
        return t + scalar;
    }


    MindTensor operator*(const f32 scalar, MindTensor& t) {
        return t * scalar;
    }
} // namespace cortex::_fw

bool MindTensor::operator!=(const MindTensor& other) const {
    return !(*this == other);
}

MindTensor &MindTensor::operator=(const MindTensor &other) {
    if (this == &other) return *this;

    this->m_shape           = other.m_shape;
    this->m_stride          = other.m_stride;
    this->m_offset          = other.m_offset;
    this->m_dev             = other.m_dev;
    this->m_grad_flag       = other.m_grad_flag;

    this->storage_          = std::make_shared<TensorStorage>(*other.storage_);
    this->flow_             = other.flow_;

    this->gradient_         = other.gradient_ ? std::make_unique<MindTensor>(*other.gradient_) : nullptr;

    return *this;
}

MindTensor &MindTensor::operator=(MindTensor &&other) noexcept {
    if (this == &other) return *this;

    this->m_shape           = std::move(other.m_shape);
    this->m_stride          = std::move(other.m_stride);
    this->m_offset          = other.m_offset;
    this->m_dev             = other.m_dev;
    this->m_grad_flag       = other.m_grad_flag;

    this->storage_          = std::move(other.storage_);
    this->flow_             = std::move(other.flow_);
    this->gradient_         = std::move(other.gradient_);

    other.m_offset          = 0;

    return *this;
}