//
// Created by muham on 18.04.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#include <CortexMind/core/Engine/AVX2/functions.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/core/Engine/CUDA/scalar.h>
    #include <CortexMind/core/Tools/utils.cuh>
    #include <CortexMind/framework/Memory/transform.hpp>
#else //#if CXM_IS_CUDA_AVAILABLE
    #include <cstring>
#endif //#if CXM_IS_CUDA_AVAILABLE #else
#include <CortexMind/framework/Gradient/operations.hpp>
#include <CortexMind/framework/Tools/tensor_utils.hpp>
#include <CortexMind/runtime/rand.cuh>
#include <random>
#include <type_traits>

using namespace cortex::_fw::meta;
using namespace cortex::_fw::runtime;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

MindTensor::MindTensor() : m_grad_flag(false), matrix(deviceType::host), reduction_ops(deviceType::host), scalar(deviceType::host), wise(deviceType::host) {
    this->storage_ = nullptr;
    this->flow_ = nullptr;
    this->gradient_ = nullptr;
}

MindTensor::MindTensor(const std::vector<i64> &shape, const deviceType device, const bool requires_grad) : m_grad_flag(requires_grad), matrix(device), reduction_ops(device), scalar(device), wise(device) {
    this->storage_ = std::make_shared<TensorStorage>(compute_numel(shape), device);

    this->storage_->shape = shape;
    this->storage_->stride = compute_stride(shape);
    this->storage_->offset = 0;

    if (this->m_grad_flag) {
        this->gradient_ = std::make_unique<MindTensor>(this->storage_->shape, this->storage_->device());
        this->gradient_->zero();
    }
}

MindTensor::MindTensor(const std::initializer_list<i64> shape, const deviceType device, const bool requires_grad) : MindTensor(std::vector(shape), device, requires_grad) {}

MindTensor::MindTensor(const TensorStorage &tensor_storage, const bool requires_grad) : m_grad_flag(requires_grad), matrix(tensor_storage.device()), reduction_ops(tensor_storage.device()), scalar(tensor_storage.device()), wise(tensor_storage.device()) {
    this->storage_ = std::make_shared<TensorStorage>(tensor_storage);

    if (this->m_grad_flag) {
        this->gradient_ = std::make_unique<MindTensor>(this->storage_->shape, this->storage_->device());
        this->gradient_->zero();
    }
}

MindTensor::MindTensor(const std::vector<i64> &shape, const f32 *data, const deviceType device, const bool requires_grad) : m_grad_flag(requires_grad), matrix(device), reduction_ops(device), scalar(device), wise(device) {
    this->storage_ = std::make_shared<TensorStorage>(compute_numel(shape), device);

    this->storage_->shape = shape;
    this->storage_->stride = compute_stride(shape);
    this->storage_->offset = 0;

    #if CXM_IS_CUDA_AVAILABLE
        if (this->storage_->device() == deviceType::host) {
            transform<f32>::copy_h2h(this->storage_->data(), data, compute_numel(shape));
        }
        if (this->storage_->device() == deviceType::cuda) {
            f32* data_gpu_clon = nullptr;
            transform<f32>::upload(data_gpu_clon, data, compute_numel(shape));
            transform<f32>::copy_d2d(this->storage_->data(), data_gpu_clon, compute_numel(shape));
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        std::memcpy(this->storage_->data(), data, compute_numel(shape) * sizeof(f32));
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    if (this->m_grad_flag) {
        this->gradient_ = std::make_unique<MindTensor>(this->shape(), this->device());
        this->gradient_->zero();
    }
}

MindTensor::MindTensor(const TensorStorage &storage, const TensorStorage &grad_storage, const std::shared_ptr<GradientFlow>& gradient_flow) : m_grad_flag(true), matrix(storage.device()), reduction_ops(storage.device()), scalar(storage.device()), wise(storage.device()) {
    this->storage_ = std::make_shared<TensorStorage>(storage);
    this->gradient_ = std::make_unique<MindTensor>(grad_storage);
    this->flow_ = gradient_flow;
}

MindTensor::MindTensor(std::shared_ptr<TensorStorage> tensor_storage, const std::shared_ptr<TensorStorage>& grad_storage, std::shared_ptr<GradientFlow> gradient_flow) : m_grad_flag(true), matrix(tensor_storage->device()), reduction_ops(tensor_storage->device()), scalar(tensor_storage->device()), wise(tensor_storage->device()) {
    this->storage_ = std::move(tensor_storage);

    this->gradient_ = std::make_unique<MindTensor>();
    this->gradient_->storage_ = grad_storage;
    this->gradient_->m_grad_flag = false;

    this->flow_ = std::move(gradient_flow);
}

MindTensor::MindTensor(const MindTensor &other) : m_grad_flag(other.m_grad_flag), matrix(other.device()), reduction_ops(other.device()), scalar(other.device()), wise(other.device()) {
    this->storage_ = other.storage_;

    this->flow_ = other.flow_;

    if (this->m_grad_flag) {
        this->gradient_ = std::make_unique<MindTensor>(*other.gradient_);
        this->gradient_->zero();
    }
}

MindTensor::MindTensor(MindTensor &&other) noexcept : m_grad_flag(other.m_grad_flag), matrix(other.device()), reduction_ops(other.device()), scalar(other.device()), wise(other.device()) {
    this->storage_ = std::move(other.storage_);

    this->flow_ = std::move(other.flow_);

    if (this->m_grad_flag) {
        this->gradient_ = std::move(other.gradient_);
    }
}

MindTensor::~MindTensor() = default;

f32 *MindTensor::get() {
    return this->storage_->data();
}

const f32 *MindTensor::get() const {
    return this->storage_->data();
}

const std::vector<i64> &MindTensor::shape() const {
    return this->storage_->shape;
}

bool MindTensor::isGradRequired() const {
    return this->m_grad_flag;
}

deviceType MindTensor::device() const {
    return this->storage_->device();
}

bool MindTensor::empty() {
    return this->storage_->isEmpty();
}

size_t MindTensor::len() const {
    return compute_numel(this->storage_->shape);
}

size_t MindTensor::ndim() const {
    return this->storage_->shape.size();
}

bool MindTensor::empty() const {
    return this->storage_->isEmpty();
}

f32 MindTensor::mean() const {
    return this->reduction_ops.mean(this->storage_.get(), this->len());
}

f32 MindTensor::variance() const {
    return this->reduction_ops.mean(this->storage_.get(), this->len());
}

f32 MindTensor::stdv() const {
    return this->reduction_ops.stdv(this->storage_.get(), this->len());
}

f32 MindTensor::max() const {
    return this->reduction_ops.max(this->storage_.get(), this->len());
}

f32 MindTensor::min() const {
    return this->reduction_ops.min(this->storage_.get(), this->len());
}

void MindTensor::ones() const {
    if (this->storage_->device() == deviceType::host) {
        size_t i = 0;
        for (; i + 8 <= this->len(); i += 8) {
            avx2::storeu(this->storage_->data() + i, avx2::set1(1.0f));
        }
        for (; i < this->len(); ++i) {
            this->storage_->data()[i] = 1.0f;
        }
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::memset<f32>(this->storage_->data(), 1, this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void MindTensor::zero() const {
    if (this->storage_->device() == deviceType::host) {
        size_t i = 0;
        for (; i + 8 <= this->len(); i += 8) {
            avx2::storeu(this->storage_->data() + i, avx2::set1(0.0f));
        }
        for (; i < this->len(); ++i) {
            this->storage_->data()[i] = 0.0f;
        }
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::memset<f32>(this->storage_->data(), 0, this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void MindTensor::fill(const f32 value) const {
    if (this->storage_->device() == deviceType::host) {
        size_t i = 0;
        for (; i + 8 <= this->len(); i += 8) {
            avx2::storeu(this->storage_->data() + i, avx2::set1(value));
        }
        for (; i < this->len(); ++i) {
            this->storage_->data()[i] = value;
        }
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
                cuda::memset<f32>(this->storage_->data(), value, this->len());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void MindTensor::rand(const f32 min, const f32 max) const {
    if (this->storage_->device() == deviceType::host) {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution dist(min, max);
        for (size_t i = 0; i < this->len(); ++i)
            this->storage_->data()[i] = dist(rng);
    }
    if (this->storage_->device() == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            const size_t count  = this->len();
            const size_t padded = count % 2 == 0 ? count : count + 1;

            CXM_ASSERT(
                curandGenerateUniform(
                    runtime::RandEngine::instance().generator,
                    this->storage_->data(),
                    padded
                ) == CURAND_STATUS_SUCCESS,
                "cortex::_fw::MindTensor::rand()",
                "curandGenerateUniform() failed"
            );

            if (min != 0.0f || max != 1.0f) {
                cuda::ScalarKernel::mul(this->storage_->data(), max - min,
                                        this->storage_->data(), count);
                cuda::ScalarKernel::add(this->storage_->data(), min,
                                        this->storage_->data(), count);
            }
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void MindTensor::backward() const {
    CXM_ASSERT(this->m_grad_flag, "cortex::_fw::MindTensor::backward()", "requires_grad is false");
    CXM_ASSERT(this->gradient_ != nullptr, "cortex::_fw::MindTensor::backward()", "gradient is not initialized");
    CXM_ASSERT(this->flow_ != nullptr, "cortex::_fw::MindTensor::backward()", "no gradient function attached");

    if (this->len() == 1) {
        this->gradient_->ones();
    }

    this->flow_->backward(this->gradient_.get());
}

void MindTensor::backward(MindTensor &other) const {
    CXM_ASSERT(this->flow_ != nullptr, "cortex::_fw::MindTensor::backward()", "no gradient function attached | The one who I fucked his blind eye");
    this->flow_->backward(&other);
}

void MindTensor::set_flow(const std::shared_ptr<GradientFlow> &_flow) {
    this->flow_ = _flow;
}

void MindTensor::set_grad(const MindTensor &_grad) {
    this->gradient_ = std::make_unique<MindTensor>(_grad);
}

MindTensor MindTensor::to(const deviceType &d_type) {
    this->storage_->setDevice(d_type);

    this->matrix.SetDevice(d_type);
    this->scalar.SetDevice(d_type);
    this->reduction_ops.SetDevice(d_type);
    this->wise.SetDevice(d_type);

    return *this;
}

MindTensor MindTensor::dot(const MindTensor& other) {
    CXM_ASSERT(this->storage_->shape.size() == 2 && other.storage_->shape.size() == 2,
        "cortex::_fw::MindTensor::dot()", "Both tensors must be 2D");
    CXM_ASSERT(this->storage_->shape[1] == other.storage_->shape[0],
        "cortex::_fw::MindTensor::dot()", "Inner dimensions must match");
    CXM_ASSERT(this->device() == other.device(),
        "cortex::_fw::MindTensor::dot()", "Devices do not match");

    const i64 M = this->storage_->shape[0];
    const i64 K = this->storage_->shape[1];
    const i64 N = other.storage_->shape[1];

    MindTensor output({M, N}, this->device(), this->m_grad_flag && other.m_grad_flag);

    this->matrix.matmul(this->storage_.get(), other.storage_.get(), output.storage_.get(), M, K, N);

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::dot>(this->storage_, other.storage_, this->gradient_->storage_, other.gradient_->storage_, this->flow_, other.flow_);
    }

    return output;
}

MindTensor MindTensor::pow(const f32 exp) {
    MindTensor output(this->storage_->shape, this->storage_->device(), this->m_grad_flag);

    this->wise.pow(this->storage_.get(), exp, output.storage_.get(), this->len());

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::pow>(this->storage_, this->gradient_->storage_, this->flow_, exp);
    }

    return output;
}

MindTensor MindTensor::sqrt() {
    MindTensor output(this->storage_->shape, this->storage_->device(), this->m_grad_flag);

    this->wise.sqrt(this->storage_.get(), output.storage_.get(), this->len());

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::pow>(this->storage_, this->gradient_->storage_, this->flow_, 0.5f);
    }

    return output;
}

MindTensor MindTensor::log() {
    MindTensor output(this->storage_->shape, this->storage_->device(), this->m_grad_flag);

    this->wise.log(this->storage_.get(), output.storage_.get(), this->len());

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::log>(this->storage_, this->gradient_->storage_, this->flow_);
    }

    return output;
}

MindTensor MindTensor::exp() {
    MindTensor output(this->storage_->shape, this->storage_->device(), this->m_grad_flag);

    this->wise.exp(this->storage_.get(), output.storage_.get(), this->len());

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::exp>(this->storage_, this->gradient_->storage_, this->flow_);
    }

    return output;
}

MindTensor MindTensor::transpose() const {
    CXM_ASSERT(this->ndim() == 2, "cortex::_fw::MindTensor::transpose()", "Only 2D tensors supported");

    MindTensor output({this->storage_->shape[1], this->storage_->shape[0]}, this->device(), this->m_grad_flag);
    output.storage_ = this->storage_;
    output.storage_->shape  = {this->storage_->shape[1], this->storage_->shape[0]};
    output.storage_->stride = {this->storage_->stride[0], this->storage_->stride[1]}; // ters çevir
    output.storage_->offset = this->storage_->offset;

    return output;
}

MindTensor MindTensor::sum() const {
    MindTensor output({1}, this->device(), this->m_grad_flag);

    const f32 result = this->reduction_ops.sum(this->storage_.get(), this->len());
    output.fill(result);

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::sum>(this->storage_, this->gradient_->storage_, this->flow_);
    }

    return output;
}

MindTensor &MindTensor::grad() {
    CXM_ASSERT(this->m_grad_flag, "cortex::_fw::MindTensor::grad()", "Gradient hasn't activated");
    return *this->gradient_;
}

const MindTensor &MindTensor::grad() const {
    CXM_ASSERT(this->m_grad_flag, "cortex::_fw::MindTensor::grad()", "Gradient hasn't activated");
    return *this->gradient_;
}