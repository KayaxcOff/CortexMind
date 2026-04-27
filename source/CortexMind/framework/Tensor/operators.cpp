//
// Created by muham on 18.04.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#include <CortexMind/core/Engine/AVX2/matrix.hpp>
#include <CortexMind/core/Engine/AVX2/scalar.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/core/Engine/CUDA/scalar.h>
    #include <CortexMind/core/Engine/CUDA/matrix.h>
    #include <CortexMind/framework/Memory/transform.hpp>
#else //#if CXM_IS_CUDA_AVAILABLE
    #include <cstring>
#endif //#if CXM_IS_CUDA_AVAILABLE #else
#include <CortexMind/framework/Gradient/operations.hpp>
#include <CortexMind/framework/Tools/err.hpp>
#include <iostream>
#include <functional>

using namespace cortex::_fw::meta;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

MindTensor MindTensor::operator+(const MindTensor &other) const {
    CXM_ASSERT(this->ndim() == other.ndim(), "cortex::_fw::MindTensor::operator+", "Shapes mismatch");
    CXM_ASSERT(this->device() == other.device(), "cortex::_fw::MindTensor::operator+", "Devices mismatch");

    const bool req_grad = this->m_grad_flag || other.m_grad_flag;
    MindTensor output(this->storage_->shape, this->storage_->device(), req_grad);

    if (this->storage_->device() == deviceType::host) {
        avx2::matrix_t::add(this->get(), other.get(), output.get(), this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::Matrix::add(this->get(), other.get(), output.get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<addition>(this->storage_, other.storage_, this->gradient_->storage_, other.gradient_->storage_, this->flow_, other.flow_);
    }

    return output;
}

MindTensor MindTensor::operator-(const MindTensor &other) const {
    CXM_ASSERT(this->ndim() == other.ndim(), "cortex::_fw::MindTensor::operator-", "Shapes mismatch");
    CXM_ASSERT(this->device() == other.device(), "cortex::_fw::MindTensor::operator-", "Devices mismatch");

    const bool req_grad = this->m_grad_flag || other.m_grad_flag;
    MindTensor output(this->storage_->shape, this->storage_->device(), req_grad);

    if (this->storage_->device() == deviceType::host) {
        avx2::matrix_t::sub(this->get(), other.get(), output.get(), this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::Matrix::sub(this->get(), other.get(), output.get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<subtraction>(this->storage_, other.storage_, this->gradient_->storage_, other.gradient_->storage_, this->flow_, other.flow_);
    }

    return output;
}

MindTensor MindTensor::operator*(const MindTensor &other) const {
    CXM_ASSERT(this->ndim() == other.ndim(), "cortex::_fw::MindTensor::operator*", "Shapes mismatch");
    CXM_ASSERT(this->device() == other.device(), "cortex::_fw::MindTensor::operator*", "Devices mismatch");

    MindTensor output(this->storage_->shape, this->storage_->device(), this->m_grad_flag && other.m_grad_flag);

    if (this->storage_->device() == deviceType::host) {
        avx2::matrix_t::mul(this->get(), other.get(), output.get(), this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::Matrix::mul(this->get(), other.get(), output.get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<multiply>(this->storage_, other.storage_, this->gradient_->storage_, other.gradient_->storage_, this->flow_, other.flow_);
    }

    return output;
}

MindTensor MindTensor::operator/(const MindTensor &other) const {
    CXM_ASSERT(this->ndim() == other.ndim(), "cortex::_fw::MindTensor::operator/", "Shapes mismatch");
    CXM_ASSERT(this->device() == other.device(), "cortex::_fw::MindTensor::operator/", "Devices mismatch");

    MindTensor output(this->storage_->shape, this->storage_->device(), this->m_grad_flag && other.m_grad_flag);

    if (this->storage_->device() == deviceType::host) {
        avx2::matrix_t::div(this->get(), other.get(), output.get(), this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
                cuda::Matrix::div(this->get(), other.get(), output.get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<division>(this->storage_, other.storage_, this->gradient_->storage_, other.gradient_->storage_, this->flow_, other.flow_);
    }

    return output;
}

MindTensor MindTensor::operator+=(const MindTensor &other) {
    CXM_ASSERT(this->ndim() == other.ndim(), "cortex::_fw::MindTensor::operator+=", "Shapes mismatch");
    CXM_ASSERT(this->device() == other.device(), "cortex::_fw::MindTensor::operator+=", "Devices mismatch");

    if (this->storage_->device() == deviceType::host) {
        avx2::matrix_t::add(this->get(), other.get(), this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::Matrix::add(this->get(), other.get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }

    return *this;
}

MindTensor MindTensor::operator-=(const MindTensor &other) {
    CXM_ASSERT(this->ndim() == other.ndim(), "cortex::_fw::MindTensor::operator-=", "Shapes mismatch");
    CXM_ASSERT(this->device() == other.device(), "cortex::_fw::MindTensor::operator-=", "Devices mismatch");

    if (this->storage_->device() == deviceType::host) {
        avx2::matrix_t::sub(this->get(), other.get(), this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::Matrix::sub(this->get(), other.get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }

    return *this;
}

MindTensor MindTensor::operator*=(const MindTensor &other) {
    CXM_ASSERT(this->ndim() == other.ndim(), "cortex::_fw::MindTensor::operator*=", "Shapes mismatch");
    CXM_ASSERT(this->device() == other.device(), "cortex::_fw::MindTensor::operator*=", "Devices mismatch");

    if (this->storage_->device() == deviceType::host) {
        avx2::matrix_t::mul(this->get(), other.get(), this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::Matrix::mul(this->get(), other.get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }

    return *this;
}

MindTensor MindTensor::operator/=(const MindTensor &other) {
    CXM_ASSERT(this->ndim() == other.ndim(), "cortex::_fw::MindTensor::operator/=", "Shapes mismatch");
    CXM_ASSERT(this->device() == other.device(), "cortex::_fw::MindTensor::operator/=", "Devices mismatch");

    if (this->storage_->device() == deviceType::host) {
        avx2::matrix_t::div(this->get(), other.get(), this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::Matrix::div(this->get(), other.get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }

    return *this;
}

MindTensor MindTensor::operator+(const f32 value) const {
    MindTensor output(this->storage_->shape, this->storage_->device(), this->m_grad_flag);

    if (this->storage_->device() == deviceType::host) {
        avx2::ScalarOp::add(this->get(), value, output.get(), this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::add(this->get(), value, output.get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<scalar_additive>(this->storage_, this->gradient_->storage_, this->flow_);
    }

    return output;
}

MindTensor MindTensor::operator-(const f32 value) const {
    MindTensor output(this->storage_->shape, this->storage_->device(), this->m_grad_flag);

    if (this->storage_->device() == deviceType::host) {
        avx2::ScalarOp::sub(this->get(), value, output.get(), this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::sub(this->get(), value, output.get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<scalar_additive>(this->storage_, this->gradient_->storage_, this->flow_);
    }

    return output;
}

MindTensor MindTensor::operator*(const f32 value) const {
    MindTensor output(this->storage_->shape, this->storage_->device(), this->m_grad_flag);

    if (this->storage_->device() == deviceType::host) {
        avx2::ScalarOp::mul(this->get(), value, output.get(), this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::mul(this->get(), value, output.get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<scalar_multiply>(this->storage_, this->gradient_->storage_, this->flow_, value);
    }

    return output;
}

MindTensor MindTensor::operator/(const f32 value) const {
    MindTensor output(this->storage_->shape, this->storage_->device(), this->m_grad_flag);

    if (this->storage_->device() == deviceType::host) {
        avx2::ScalarOp::div(this->get(), value, output.get(), this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::div(this->get(), value, output.get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<scalar_multiply>(this->storage_, this->gradient_->storage_, this->flow_, (1 / value));
    }

    return output;
}

MindTensor MindTensor::operator+=(const f32 value) {
    if (this->storage_->device() == deviceType::host) {
        avx2::ScalarOp::add(this->get(), value, this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::add(this->get(), value, this->get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
    return *this;
}

MindTensor MindTensor::operator-=(const f32 value) {
    if (this->storage_->device() == deviceType::host) {
        avx2::ScalarOp::sub(this->get(), value, this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::sub(this->get(), value, this->get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
    return *this;
}

MindTensor MindTensor::operator*=(const f32 value) {
    if (this->storage_->device() == deviceType::host) {
        avx2::ScalarOp::mul(this->get(), value, this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::mul(this->get(), value, this->get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
    return *this;
}

MindTensor MindTensor::operator/=(const f32 value) {
    if (this->storage_->device() == deviceType::host) {
        avx2::ScalarOp::div(this->get(), value, this->len());
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::div(this->get(), value, this->get(), this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
    return *this;
}

MindTensor &MindTensor::operator=(const MindTensor &other) {
    if (this == &other) {
        return *this;
    }

    this->storage_ = other.storage_;
    this->m_grad_flag = other.m_grad_flag;
    this->flow_ = other.flow_;

    if (other.m_grad_flag) {
        this->gradient_ = std::make_unique<MindTensor>(*other.gradient_);
    }

    return *this;
}

MindTensor &MindTensor::operator=(MindTensor &&other) noexcept {
    this->storage_ = std::move(other.storage_);
    this->m_grad_flag = other.m_grad_flag;
    this->flow_ = std::move(other.flow_);
    this->gradient_ = std::move(other.gradient_);

    other.storage_ = nullptr;
    other.flow_ = nullptr;

    return *this;
}

namespace cortex::_fw {
    std::ostream& operator<<(std::ostream& os, const MindTensor& tensor) {
        const auto& shape = tensor.storage_->shape;
        const size_t numel = tensor.len();

        std::vector<f32> host_data(numel);

        if (tensor.device() == sys::deviceType::host) {
            std::memcpy(host_data.data(), tensor.get(), numel * sizeof(f32));
        }
        #if CXM_IS_CUDA_AVAILABLE
        else {
            transform<f32>::download(host_data.data(), tensor.get(), numel);
        }
        #endif //#if CXM_IS_CUDA_AVAILABLE

        std::function<void(size_t, size_t, size_t)> print_dim =
            [&](const size_t dim, const size_t offset, const size_t indent) {
                if (dim == shape.size() - 1) {
                    os << "[";
                    for (i64 i = 0; i < shape[dim]; ++i) {
                        os << host_data[offset + i];
                        if (i < shape[dim] - 1) os << ", ";
                    }
                    os << "]";
                } else {
                    os << "[";
                    const size_t stride = tensor.storage_->stride[dim];
                    for (i64 i = 0; i < shape[dim]; ++i) {
                        if (i > 0) {
                            os << ",\n";
                            for (size_t s = 0; s <= indent; ++s) os << " ";
                        }
                        print_dim(dim + 1, offset + i * stride, indent + 1);
                    }
                    os << "]";
                }
        };

        print_dim(0, tensor.storage_->offset, 0);
        os << "\n";

        return os;
    }
} //namespace cortex::_fw