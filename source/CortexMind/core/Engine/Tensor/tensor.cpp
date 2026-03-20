//
// Created by muham on 15.03.2026.
//

#include "CortexMind/core/Engine/Tensor/tensor.hpp"
#include <CortexMind/core/Engine/AVX2/matrix.hpp>
#include <CortexMind/core/Engine/AVX2/scalar.hpp>
#include <CortexMind/core/Engine/CUDA/activation.cuh>
#include <CortexMind/core/Engine/CUDA/matrix.cuh>
#include <CortexMind/core/Engine/CUDA/rand.cuh>
#include <CortexMind/core/Engine/CUDA/reduce.cuh>
#include <CortexMind/core/Engine/Memory/transform.cuh>
#include <CortexMind/core/Graph/ops.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>

using namespace cortex::_fw;
using namespace cortex::_fw::sys;

MindTensor::MindTensor() : m_dev(dev::host), m_offset(0), m_grad_flag(false) {
    this->storage_  = nullptr;
    this->flow_ = nullptr;
    this->gradient_ = nullptr;
}

MindTensor::MindTensor(const std::vector<i64> &shape, const dev d, const bool requires_grad) : m_dev(d), m_shape(shape), m_offset(0), m_grad_flag(requires_grad) {
    this->storage_ = std::make_shared<TensorStorage>(this->numel(), this->m_dev);
    this->m_stride = compute_stride(this->m_shape);

    if (this->m_grad_flag) {
        this->gradient_ = std::make_unique<MindTensor>(this->m_shape, this->m_dev);
        this->gradient_->zero();
    }
}

MindTensor::MindTensor(const std::initializer_list<i64> &shape, const dev d, const bool requires_grad) : MindTensor(std::vector(shape), d, requires_grad) {}

MindTensor::MindTensor(const std::vector<i64> &shape, const f32 *data, const dev d, const bool requires_grad) : m_dev(d), m_shape(shape), m_offset(0), m_grad_flag(requires_grad) {
    this->storage_ = std::make_shared<TensorStorage>(this->numel(), this->m_dev);
    this->m_stride = compute_stride(this->m_shape);

    if (this->storage_->is_cpu()) {
        transform<f32>::copy_h2h(data, this->storage_->data(), this->storage_->size());
    } else if (this->storage_->is_gpu()) {
        transform<f32>::copy_d2d(data, this->storage_->data(), this->storage_->size());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::MindTensor()", "Invalid device");
    }

    if (this->m_grad_flag) {
        this->gradient_ = std::make_unique<MindTensor>(this->m_shape, this->m_dev);
        this->gradient_->zero();
    }
}

MindTensor::MindTensor(const MindTensor &other) : m_dev(other.m_dev), m_shape(other.m_shape), m_stride(other.m_stride), m_offset(other.m_offset), m_grad_flag(other.m_grad_flag) {
    this->storage_  = other.storage_;
    this->flow_ = other.flow_;
}

MindTensor::MindTensor(MindTensor &&other) noexcept : m_dev(other.m_dev), m_shape(std::move(other.m_shape)), m_stride(std::move(other.m_stride)), m_offset(other.m_offset), m_grad_flag(other.m_grad_flag) {
    this->storage_  = std::move(other.storage_);
    this->flow_ = std::move(other.flow_);

    if (this->m_grad_flag) {
        this->gradient_ = std::move(other.gradient_);
    }
}

MindTensor::~MindTensor() = default;

f32 *MindTensor::get() {
    CXM_WARN(!this->storage_->is_gpu(), "cortex::_fw::MindTensor::get()", "Data is on GPU device""\n""You shouldn't try index access to tensor");
    return this->storage_->data() + this->m_offset;
}

f32 *MindTensor::get() const {
    CXM_WARN(!this->storage_->is_gpu(), "cortex::_fw::MindTensor::get()", "Data is on GPU device""\n""You shouldn't try index access to tensor");
    return this->storage_->data() + this->m_offset;
}

const std::vector<i64> &MindTensor::shape() const {
    return this->m_shape;
}

const std::vector<i64> &MindTensor::stride() const {
    return this->m_stride;
}

size_t MindTensor::numel() const noexcept {
    return compute_numel(this->m_shape);
}

i64 MindTensor::ndim() const noexcept {
    return static_cast<i64>(this->m_shape.size());
}

bool MindTensor::grad_required() const noexcept {
    return this->m_grad_flag;
}

bool MindTensor::empty() const noexcept {
    return this->storage_ == nullptr || this->storage_->isEmpty();
}

bool MindTensor::contiguous() const noexcept {
    return is_contiguous(this->m_shape, this->m_stride);
}

dev MindTensor::device() const noexcept {
    return this->m_dev;
}

f32 MindTensor::mean() const {
    f32 output = 0.0f;

    if (this->m_dev == dev::host) {
        output = avx2::ScalarOp::mean(this->get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::reduce_t reduce;
        output = reduce.mean(this->get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::mean()", "Invalid device");
    }
    return output;
}

f32 MindTensor::variance() const {
    f32 output = 0.0f;

    if (this->m_dev == dev::host) {
        output = avx2::ScalarOp::var(this->get(), this->mean(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::reduce_t reduce;
        output = reduce.var(this->get(), this->numel());
    }  else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::variance()", "Invalid device");
    }
    return output;
}

f32 MindTensor::std_dev() const {
    f32 output = 0.0f;

    if (this->m_dev == dev::host || this->m_dev == dev::cuda) {
        output = std::sqrt(this->variance());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::std_dev()", "Invalid device");
    }
    return output;
}

f32 MindTensor::max() const {
    f32 output = 0.0f;

    if (this->m_dev == dev::host) {
        output = avx2::ScalarOp::max(this->get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::reduce_t reduce;
        output = reduce.hmax(this->get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::max()", "Invalid device");
    }
    return output;
}

f32 MindTensor::min() const {
    f32 output = 0.0f;

    if (this->m_dev == dev::host) {
        output = avx2::ScalarOp::min(this->get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::reduce_t reduce;
        output = reduce.hmin(this->get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::min()", "Invalid device");
    }
    return output;
}

f32 MindTensor::norm() const {
    f32 output = 0.0f;

    if (this->m_dev == dev::host) {
        output = avx2::ScalarOp::norm(this->get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::reduce_t reduce;
        output = reduce.norm(this->get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::norm()", "Invalid device");
    }
    return output;
}

f32 MindTensor::sum_all() const {
    f32 output = 0.0f;

    if (this->m_dev == dev::host) {
        output = avx2::ScalarOp::sum(this->get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::reduce_t reduce;
        output = reduce.sum(this->get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::sum_all()", "Invalid device");
    }
    return output;
}

void MindTensor::backward() const {
    CXM_ASSERT(this->m_grad_flag, "cortex::_fw::MindTensor::backward()", "Grad flag is false so you can't use gradient");
    CXM_ASSERT(this->gradient_.get(), "cortex::_fw::MindTensor::backward()", "Gradient is null");

    if (this->numel() == 1) this->gradient_->ones();

    if (this->flow_) this->flow_->backward(*this->gradient_);
}

void MindTensor::backward(MindTensor *_grad) const {
    if (this->flow_) this->flow_->backward(*_grad);
}

void MindTensor::print() const {
    if (this->numel() == 0) {
        std::cerr << "Tensor is empty\n";
        return;
    }

    constexpr i32 indent_step = 1;

    std::cout << std::fixed << std::setprecision(4);

    auto indent = [](const i32 n) {
        std::cout << std::string(n, ' ');
    };

    auto print_recursive = [&](auto&& self, const i64 dim, const i64 offset, const i32 ind) -> void {
        if (dim == static_cast<i64>(this->m_shape.size()) - 1) {
            std::cout << "[";
            for (i64 i = 0; i < this->m_shape[dim]; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << this->storage_->data()[offset + i * this->m_stride[dim]];
            }
            std::cout << "]";
            return;
        }

        std::cout << "[";
        for (i64 i = 0; i < this->m_shape[dim]; ++i) {
            if (i > 0) {
                std::cout << ",\n";
                indent(ind + indent_step);
            }

            self(self,
                 dim + 1,
                 offset + i * this->m_stride[dim],
                 ind + indent_step);
        }
        std::cout << "]";
    };

    print_recursive(print_recursive, 0, this->m_offset, 0);
    std::cout << "\n";
}

void MindTensor::uniform(f32 min, f32 max) {
    if (min > max) std::swap(min, max);
    thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution dist(min, max);

    if (this->m_dev == dev::host) {
        f32* ptr = this->get();
        const size_t n = this->numel();
        for (size_t i = 0; i < n; ++i)
            ptr[i] = dist(gen);
    } else if (this->m_dev == dev::cuda) {
        cuda::rand::uniform(this->get(), min, max, this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::uniform()", "Invalid device");
    }
}

void MindTensor::zero() const {
    if (this->m_dev == dev::host) {
        avx2::matrix_t::fill(this->get(), 0, this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::matrix_t::fill(this->get(), 0.0f, this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::zero()", "Invalid device");
    }
}

void MindTensor::ones() const {
    if (this->m_dev == dev::host) {
        avx2::matrix_t::fill(this->get(), 1, this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::matrix_t::fill(this->get(), 1.0f, this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::ones()", "Invalid device");
    }
}

void MindTensor::fill(const f32 val) const {
    if (this->m_dev == dev::host) {
        avx2::matrix_t::fill(this->get(), val, this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::matrix_t::fill(this->get(), val, this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::fill()", "Invalid device");
    }
}

void MindTensor::require_grad(const bool _require_grad) {
    this->m_grad_flag = _require_grad;

    if (this->m_grad_flag && !this->gradient_) {
        this->gradient_ = std::make_unique<MindTensor>(this->m_shape);
        this->gradient_->zero();
    }
}

void MindTensor::set_grad(std::unique_ptr<MindTensor> _grad) {
    CXM_ASSERT(this->m_grad_flag, "cortex::_fw::MindTensor::set_grad()", "Gradient flag is false so you can't use gradient");
    this->gradient_ = std::move(_grad);
}

void MindTensor::set_grad(const MindTensor &grad) {
    CXM_ASSERT(this->m_grad_flag, "cortex::_fw::MindTensor::set_grad()", "Gradient flag is false so you can't use gradient");
    this->gradient_ = std::make_unique<MindTensor>(grad);
}

void MindTensor::set_flow(std::shared_ptr<meta::GradientFlow> _flow) {
    this->flow_ = std::move(_flow);
}

MindTensor MindTensor::to(const dev _device) {
    this->m_dev = _device;

    this->storage_->to(this->m_dev);

    return *this;
}

MindTensor MindTensor::flat() const {
    i64 batch = this->m_shape[0];

    i64 feat = 1;
    for (size_t i = 1; i < this->m_shape.size(); ++i) feat *= this->m_shape[i];

    if (!is_contiguous(this->m_shape, this->m_stride)) {
        CXM_ASSERT(this->m_dev == dev::host, "cortex::_fw::MindTensor::flatten()", "Non-contiguous GPU tensor flatten not yet supported.");
        const MindTensor output = this->copy();
        return output.flat();
    }

    MindTensor output({batch, feat}, this->m_dev, this->m_grad_flag);
    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;
    return output;
}

MindTensor MindTensor::matmul(MindTensor other) {
    CXM_ASSERT(this->m_stride.size() == 2 && other.m_stride.size() == 2, "cortex::_fw::MindTensor::matmul()", "Both tensors must be 2D for matrix multiplication");
    CXM_ASSERT(this->m_shape[1] == other.m_shape[0], "cortex::_fw::MindTensor::matmul()", "Inner dimensions must match for matrix multiplication");
    CXM_ASSERT(this->m_dev == other.m_dev, "cortex::_fw::MindTensor::matmul()", "Both tensors must be on the same device.");

    MindTensor output({this->m_shape[0], other.m_shape[1]}, this->m_dev, this->m_grad_flag && other.m_grad_flag);

    if (this->m_dev == dev::host) {
        avx2::matrix_t::matmul(this->get(), other.get(), output.get(), static_cast<size_t>(this->m_shape[0]), static_cast<size_t>(this->m_shape[1]), static_cast<size_t>(other.m_shape[1]));
    } else if (this->m_dev == dev::cuda) {
        cuda::matrix_t::matmul(this->get(), other.get(), output.get(), static_cast<size_t>(this->m_shape[0]), static_cast<size_t>(this->m_shape[1]), static_cast<size_t>(other.m_shape[1]));
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::matmul()", "Invalid device");
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::matmul>(this, &other);
    }

    return output;
}
// can be on CUDA
MindTensor MindTensor::permute(const std::vector<i64> &axes) const {
    CXM_ASSERT(this->m_shape.size() == axes.size(), "cortex::_fw::MindTensor::permute()", "Number of axes must match tensor dimensions");
    for (const auto item : axes) CXM_ASSERT(item >= 0 && item < static_cast<int64_t>(this->m_shape.size()), "cortex::_fw::MindTensor::permute()", "Invalid axis index");

    std::vector seen(axes.size(), false);
    for (const auto item : axes) {
        CXM_ASSERT(!seen[item], "cortex::_fw::MindTensor::permute()", "Duplicate axis in permutation.");
        seen[item] = true;
    }

    std::vector<i64> _shape(this->m_shape.size());
    std::vector<i64> _stride(this->m_shape.size());

    for (size_t i = 0; i < axes.size(); ++i) {
        _shape[i] = this->m_shape[axes[i]];
        _stride[i] = this->m_stride[axes[i]];
    }

    MindTensor output(_shape, this->m_dev, this->m_grad_flag);

    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;
    output.m_stride = _stride;
    return output;
}

MindTensor MindTensor::copy() const {
    MindTensor output(this->m_shape, this->m_dev, this->m_grad_flag);

    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;

    return output;
}

MindTensor MindTensor::detach() const {
    MindTensor output(this->m_shape, this->m_dev, this->m_grad_flag);

    output.m_offset  = this->m_offset;
    output.m_stride  = this->m_stride;
    output.storage_  = this->storage_;

    return output;
}

MindTensor MindTensor::reshape(const std::vector<i64> &shape) const {
    CXM_ASSERT(is_contiguous(this->m_shape, this->m_stride), "cortex::_fw::MindTensor::reshape()", "Tensor must be contiguous");

    CXM_ASSERT(compute_numel(shape) == this->numel(), "cortex::_fw::MindTensor::reshape()", "Total elements must be same");

    MindTensor output(shape, this->m_dev, this->m_grad_flag);

    output.storage_ = this->storage_;
    output.m_offset = m_offset;
    return output;
}

MindTensor MindTensor::sqrt() {
    MindTensor output(this->m_shape, this->m_dev, this->m_grad_flag);

    if (this->m_dev == dev::host) {
        avx2::matrix_t::sqrt(this->get(), output.get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::matrix_t::sqrt(this->get(), output.get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::sqrt()", "Invalid device");
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::sqrt>(this);
    }

    return output;
}

MindTensor MindTensor::pow(const f32 value) {
    MindTensor output(this->m_shape, this->m_dev, this->m_grad_flag);

    if (this->m_dev == dev::host) {
        avx2::matrix_t::pow(this->get(), value, output.get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::matrix_t::pow(this->get(), value, output.get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::pow()", "Invalid device");
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::pow>(this, value);
    }

    return output;
}

MindTensor MindTensor::sum() {
    MindTensor output({1}, this->m_dev, this->m_grad_flag);

    if (this->m_dev == dev::host) {
        output.get()[0] = avx2::ScalarOp::sum(this->get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        cuda::reduce_t reduce;
        output.get()[0] = reduce.sum(this->get(), this->numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::sum()", "Invalid device");
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::sum_all>(this);
    }

    return output;
}

MindTensor MindTensor::sum(const i64 dim, const bool keep) {
    CXM_ASSERT(dim >= 0 && dim < this->ndim(), "cortex::_fw::MindTensor::sum()", "Invalid dimension");

    size_t outer = 1, after = 1;
    for (i64 i = 0; i < dim; ++i)         outer *= static_cast<size_t>(this->m_shape[i]);
    for (i64 i = dim + 1; i < this->ndim(); ++i) after *= static_cast<size_t>(this->m_shape[i]);
    const auto inner = static_cast<size_t>(this->m_shape[dim]);

    // output shape
    std::vector<i64> out_shape;
    for (i64 i = 0; i < this->ndim(); ++i) {
        if (i == dim) {
            if (keep) out_shape.push_back(1);
        } else {
            out_shape.push_back(this->m_shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    MindTensor output(out_shape, this->m_dev, this->m_grad_flag);
    output.zero();

    if (this->m_dev == dev::host) {
        const f32* src = this->get();
        f32* dst = output.get();
        for (size_t o = 0; o < outer; ++o) {
            for (size_t a = 0; a < after; ++a) {
                f32 acc = 0.0f;
                for (size_t inn = 0; inn < inner; ++inn)
                    acc += src[o * inner * after + inn * after + a];
                dst[o * after + a] = acc;
            }
        }
    } else if (this->m_dev == dev::cuda) {
        cuda::reduce_t reduce;
        reduce.sum_dim(this->get(), output.get(), outer, inner, after);
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::sum(dim)", "Invalid device");
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::sum_dim>(this, dim, keep);
    }

    return output;
}

MindTensor MindTensor::transpose() {
    CXM_ASSERT(this->ndim() == 2, "cortex::_fw::MindTensor::transpose()", "Tensor must be 2D.");

    MindTensor output({this->m_shape[1], this->m_shape[0]}, this->m_dev, this->m_grad_flag);
    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;
    output.m_stride = {this->m_stride[1], this->m_stride[0]};

    return output;
}

MindTensor MindTensor::exp() {
    MindTensor output(this->m_shape, this->m_dev, this->m_grad_flag);

    if (this->m_dev == dev::host) {
        avx2::matrix_t::exp(this->get(), output.get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        output = this->copy();
        cuda::activation_t::exp(output.get(), output.numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::exp()", "Invalid device");
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::exp>(this);
    }

    return output;
}

MindTensor MindTensor::log() {
    MindTensor output(this->m_shape, this->m_dev);

    if (this->m_dev == dev::host) {
        avx2::matrix_t::log(this->get(), output.get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        output = this->copy();
        cuda::activation_t::log(output.get(), output.numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::log()", "Invalid device");
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::log>(this);
    }

    return output;
}

MindTensor MindTensor::abs() {
    MindTensor output(this->m_shape, this->m_dev);

    if (this->m_dev == dev::host) {
        avx2::matrix_t::abs(this->get(), output.get(), this->numel());
    } else if (this->m_dev == dev::cuda) {
        output = this->copy();
        cuda::activation_t::abs(output.get(), output.numel());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::abs()", "Invalid device");
    }

    if (output.m_grad_flag) {
        output.flow_ = std::make_shared<meta::abs>(this);
    }

    return output;
}

MindTensor MindTensor::unsqueeze(const i64 dim) const {
    CXM_ASSERT(dim >= 0 && dim <= this->ndim(),
        "cortex::_fw::MindTensor::unsqueeze()", "Invalid dimension");

    std::vector<i64> new_shape  = this->m_shape;
    std::vector<i64> new_stride = this->m_stride;

    new_shape.insert(new_shape.begin() + dim, 1);
    new_stride.insert(new_stride.begin() + dim,
        dim < this->ndim() ? this->m_stride[dim] : 1);

    MindTensor output(new_shape, this->m_dev);
    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;
    output.m_stride = new_stride;
    return output;
}

MindTensor MindTensor::squeeze(const i64 dim) const {
    CXM_ASSERT(dim >= 0 && dim < this->ndim(),
        "cortex::_fw::MindTensor::squeeze()", "Invalid dimension");
    CXM_ASSERT(this->m_shape[dim] == 1,
        "cortex::_fw::MindTensor::squeeze()", "Dimension size must be 1 to squeeze");

    std::vector<i64> new_shape  = this->m_shape;
    std::vector<i64> new_stride = this->m_stride;

    new_shape.erase(new_shape.begin() + dim);
    new_stride.erase(new_stride.begin() + dim);

    if (new_shape.empty()) new_shape.push_back(1);
    if (new_stride.empty()) new_stride.push_back(1);

    MindTensor output(new_shape, this->m_dev);
    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;
    output.m_stride = new_stride;
    return output;
}

MindTensor MindTensor::slice(const i64 start, const i64 end) const {
    CXM_ASSERT(this->ndim() >= 1,
        "cortex::_fw::MindTensor::slice()", "Tensor must have at least 1 dimension");
    CXM_ASSERT(start >= 0 && start < this->m_shape[0],
        "cortex::_fw::MindTensor::slice()", "Invalid start index");
    CXM_ASSERT(end > start && end <= this->m_shape[0],
        "cortex::_fw::MindTensor::slice()", "Invalid end index");

    std::vector<i64> new_shape = this->m_shape;
    new_shape[0] = end - start;

    MindTensor output(new_shape, this->m_dev);
    output.storage_ = this->storage_;
    output.m_offset = this->m_offset + start * this->m_stride[0];
    output.m_stride = this->m_stride;
    return output;
}

MindTensor &MindTensor::grad() {
    CXM_ASSERT(this->gradient_.get(), "cortex::_fw::MindTensor::grad()", "Gradient is null");
    return *this->gradient_;
}

const MindTensor &MindTensor::grad() const {
    CXM_ASSERT(this->gradient_.get(), "cortex::_fw::MindTensor::grad()", "Gradient is null");
    return *this->gradient_;
}