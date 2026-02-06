//
// Created by muham on 4.02.2026.
//

#include "CortexMind/core/Engine/Tensor/tensor.hpp"
#include <CortexMind/core/Engine/AVX/matrix.hpp>
#include <CortexMind/core/Engine/AVX/funcs.hpp>
#include <CortexMind/core/Tools/error.hpp>
#include <CortexMind/core/Tools/restrict.hpp>
#include <utility>
#include <functional>
#include <iomanip>
#include <random>
#include <cmath>

using namespace cortex::_fw;

MindTensor::MindTensor() : m_shape({}), m_strides({}), m_offset(0), m_grad_flag(false) {}

MindTensor::MindTensor(std::vector<i64> shape, const bool _requires_grad) : m_shape(std::move(shape)), m_offset(0), m_grad_flag(_requires_grad) {
    this->m_strides = compute_strides(this->m_shape);

    size_t num = this->m_shape.empty() ? 0 : (this->m_strides[0] * this->m_shape[0]);

    this->storage_ = std::make_shared<TensorStorage>(num);

    if (this->m_grad_flag) {
        this->gradient_ = std::make_unique<MindTensor>(this->m_shape, false);
        this->gradient_->zero();
    }
}

MindTensor::MindTensor(const std::initializer_list<i64> shape, const bool _requires_grad) : MindTensor(std::vector(shape), _requires_grad) {}

MindTensor::MindTensor(const MindTensor &other) : m_shape(other.m_shape), m_offset(other.m_offset), m_grad_flag(other.m_grad_flag) {
    this->m_strides = other.m_strides;
    this->storage_ = std::make_shared<TensorStorage>(other.storage_->size());

    std::memcpy(storage_->data(), other.storage_->data(), other.storage_->size() * sizeof(f32));

    if (other.m_grad_flag) {
        this->gradient_ = std::make_unique<MindTensor>(*other.gradient_);
        this->gradient_->zero();
    }
}

f32 &MindTensor::at(const std::vector<i64>& indices) {
    const i64 idx = compute_offset(indices, this->m_strides);
    return this->storage_->data()[idx];
}

const f32 &MindTensor::at(const std::vector<i64>& indices) const {
    const i64 idx = compute_offset(indices, this->m_strides);
    return this->storage_->data()[idx];
}

f32 *MindTensor::get() {
    return this->storage_->data();
}

const f32 *MindTensor::get() const {
    return this->storage_->data();
}

std::vector<i64> MindTensor::shape() {
    return this->m_shape;
}

size_t MindTensor::numel() const {
    i64 output = 1;
    for (const auto& item : this->m_shape) output *= item;
    return output;
}

bool MindTensor::empty() const {
    return this->storage_->isEmpty();
}

bool MindTensor::requires_grad() const {
    return this->m_grad_flag;
}

bool MindTensor::is_contiguous() const {
    return this->m_strides == compute_strides(this->m_shape);
}

f32 MindTensor::mean() const {
    const size_t num = this->numel();
    CXM_ASSERT(num > 0, "cortex::_fw::MindTensor::mean()", "Cannot compute mean of empty tensor");
    const f32* restrict dx = this->storage_->data() + this->m_offset;

    avx2::vec8f vx = avx2::zero();

    size_t i = 0;
    for (; i + 8 < num; i += 8) {
        const avx2::vec8f vy = avx2::loadu(dx + i);
        vx = avx2::add(vx, vy);
    }
    f32 sum = 0;
    for (; i < num; ++i) sum += dx[i];

    const f32 vz = avx2::hsum(vx);
    return (vz + sum) / static_cast<f32>(num);
}

f32 MindTensor::variance() const {
    const f32 mean = this->mean();
    const size_t num = this->numel();
    const f32* restrict dx = this->storage_->data() + this->m_offset;

    avx2::vec8f vx = avx2::zero();
    const avx2::vec8f vy = avx2::set(mean);

    size_t i = 0;
    for (; i + 8 < num; i += 8) {
        const avx2::vec8f vz = avx2::loadu(dx + i);
        const avx2::vec8f vk = avx2::sub(vz, vy);
        vx = avx2::add(vx, avx2::mul(vk, vk));
    }
    f32 sum = 0;
    for (; i < num; ++i) sum += (dx[i] - mean) * (dx[i] - mean);
    return sum / static_cast<f32>(num);
}

f32 MindTensor::max() const {
    const f32* restrict dx = this->storage_->data();
    const size_t num = this->numel();

    CXM_ASSERT(dx != nullptr, "cortex::_fw::MindTensor::max()", "Tensor storage pointer is null");
    CXM_ASSERT(num > 0, "cortex::_fw::MindTensor::max()", "Tensor is empty");
    CXM_ASSERT(this->is_contiguous(), "cortex::_fw::MindTensor::max()", "Tensor must be contiguous");

    f32 output = dx[0];
    for (size_t i = 1; i < num; ++i) {
        if (dx[i] > output) output = dx[i];
    }
    return output;
}

f32 MindTensor::min() const {
    const f32* restrict dx = this->storage_->data();
    const size_t num = this->numel();

    CXM_ASSERT(dx != nullptr, "cortex::_fw::MindTensor::min()", "Tensor storage pointer is null");
    CXM_ASSERT(num > 0, "cortex::_fw::MindTensor::min()", "Tensor is empty");
    CXM_ASSERT(this->is_contiguous(), "cortex::_fw::MindTensor::min()", "Tensor must be contiguous");

    f32 output = dx[0];
    for (size_t i = 1; i < num; ++i) {
        if (dx[i] < output) output = dx[i];
    }
    return output;
}

void MindTensor::backward() {
    CXM_ASSERT(this->m_shape.empty(), "cortex::_fw::MindTensor::backward()", "Tensor shouldn't be empty for gradient");

    if (!this->gradient_) this->gradient_ = std::make_unique<MindTensor>(this->m_shape);

    if (this->numel() == 1) this->gradient_->ones();

    if (this->flow_) this->flow_->backward(*this->gradient_);
}

void MindTensor::print() const {
    if (this->m_shape.empty()) {
        std::cerr << "Tensor is empty" << std::endl;
        return;
    }

    constexpr i32 indent_step = 1;

    auto indent = [](const i32 n) {
        for (i32 i = 0; i < n; ++i) std::cout << ' ';
    };

    std::function<void(i64, i64, i32)> PrintRecursive;

    PrintRecursive = [&](const i64 dim, const i64 offset, const i32 ind) {
        if (dim == static_cast<i64>(this->m_shape.size()) - 1) {
            std::cout << "[";
            for (i64 i = 0; i < this->m_shape[dim]; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(4) << this->storage_->data()[offset + i * this->m_strides[dim]];
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

            PrintRecursive(dim + 1, offset + i * this->m_strides[dim], ind + indent_step);
        }
        std::cout << "]";
    };

    PrintRecursive(0, this->m_offset, 0);
    std::cout << "\n";
}

void MindTensor::uniform_rand(f32 min, f32 max) const {
    if (min > max) std::swap(min, max);

    thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution dist(min, max);

    const size_t num = this->numel();
    f32* restrict dx = this->storage_->data() + this->m_offset;

    for (size_t i = 0; i < num; ++i) dx[i] = dist(gen);
}

void MindTensor::zero() const {
    std::memset(this->storage_->data() + this->m_offset, 0, this->numel() * sizeof(f32));
}

void MindTensor::ones() const {
    const size_t num = this->numel();
    f32* restrict dx = this->storage_->data() + this->m_offset;

    for (size_t i = 0; i < num; ++i) dx[i] = 1.0f;
}

void MindTensor::fill(const f32 value) const {
    const size_t num = this->numel();
    f32* restrict dx = this->storage_->data() + this->m_offset;

    for (size_t i = 0; i < num; ++i) dx[i] = value;
}

void MindTensor::require_grad(const bool _require_grad) {
    this->m_grad_flag = _require_grad;

    if (this->m_grad_flag) {
        this->gradient_ = std::make_unique<MindTensor>(this->m_shape);
        this->gradient_->zero();
    }
}

void MindTensor::set_grad(std::unique_ptr<MindTensor> _grad) {
    CXM_ASSERT(this->m_grad_flag, "cortex::_fw::MindTensor::set_grad()", "Gradient flag is false");
    this->gradient_ = std::move(_grad);
}

void MindTensor::set_grad(const MindTensor &grad) {
    CXM_ASSERT(this->m_grad_flag, "cortex::_fw::MindTensor::set_grad()", "Gradient flag is false");
    this->gradient_ = std::make_unique<MindTensor>(grad);
}

void MindTensor::zero_grad() const {
    CXM_ASSERT(this->m_grad_flag, "cortex::_fw::MindTensor::set_grad()", "Gradient flag is false");
    this->gradient_->zero();
}

MindTensor MindTensor::flatten() const {
    i64 batch = this->m_shape[0];

    i64 feat = 1;
    for (size_t i = 1; i < this->m_shape.size(); ++i) feat *= this->m_shape[i];

    MindTensor output({batch, feat}, this->m_grad_flag);

    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;
    output.m_strides = compute_strides(output.m_shape);

    return output;
}

MindTensor MindTensor::matmul(const MindTensor& other) {
    CXM_ASSERT(this->m_strides.size() == 2 && other.m_strides.size() == 2, "cortex::_fw::MindTensor::matmul()", "Both tensors must be 2D for matrix multiplication");
    CXM_ASSERT(this->m_shape[1] == other.m_shape[0], "cortex::_fw::MindTensor::matmul()", "Inner dimensions must match for matrix multiplication");

    MindTensor output({this->m_shape[0], other.m_shape[1]}, this->m_grad_flag | other.m_grad_flag);

    avx2::matrix_t::matmul(
        this->storage_->data(),
        other.storage_->data(),
        output.storage_->data(),
        static_cast<size_t>(this->m_shape[0]),
        static_cast<size_t>(other.m_shape[1]),
        static_cast<size_t>(this->m_shape[1])
    );
    return output;
}

MindTensor MindTensor::permute(const std::vector<i64> &axes) const {
    CXM_ASSERT(this->m_shape.size() == axes.size(), "cortex::_fw::MindTensor::permute()", "Number of axes must match tensor dimensions");
    for (const auto item : axes) CXM_ASSERT(item >= 0 && item < static_cast<int64_t>(this->m_shape.size()), "cortex::_fw::MindTensor::permute()", "Invalid axis index");

    std::vector<i64> _shape(this->m_shape.size());
    std::vector<i64> _strides(this->m_shape.size());

    for (size_t i = 0; i < axes.size(); ++i) {
        _shape[i] = this->m_shape[axes[i]];
        _strides[i] = this->m_strides[axes[i]];
    }

    MindTensor output(_shape, this->m_grad_flag);
    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;
    output.m_strides = compute_strides(output.m_shape);
    return output;
}

MindTensor MindTensor::clone() const {
    MindTensor output(m_shape, this->m_grad_flag);
    output.m_offset = m_offset;
    output.m_strides = m_strides;
    return output;
}

MindTensor MindTensor::copy() const {
    MindTensor out(m_shape, this->m_grad_flag);
    out.storage_ = this->storage_;
    out.m_offset = m_offset;
    out.m_strides = m_strides;
    return out;
}

MindTensor MindTensor::reshape(const std::vector<i64>& shape) const {
    CXM_ASSERT(this->is_contiguous(), "reshape", "Tensor must be contiguous");

    i64 sum = 1;
    for (const auto item : shape) sum *= item;

    CXM_ASSERT(sum == this->numel(), "cortex::_fw::MindTensor::shape()", "Total elements must be same");

    MindTensor output(shape, this->m_grad_flag);

    output.storage_ = this->storage_;
    output.m_offset = m_offset;
    output.m_strides = compute_strides(output.m_shape);
    return output;
}

MindTensor MindTensor::sqrt() const {
    const size_t num = this->numel();
    const f32* restrict dx = this->storage_->data() + this->m_offset;

    MindTensor output(this->m_shape, this->m_grad_flag);

    f32* restrict dy = output.storage_->data() + output.m_offset;

    size_t i = 0;
    for (; i + 7 < num; i += 8) {
        avx2::vec8f vx = avx2::loadu(dx + i);
        vx = avx2::sqrt(vx);
        avx2::storeu(dy + i, vx);
    }
    for (; i < num; ++i) dy[i] = std::sqrt(dx[i]);
    return output;
}

MindTensor MindTensor::pow(const f32 value) const {
    const size_t num = this->numel();
    const f32* restrict dx = this->storage_->data() + this->m_offset;

    MindTensor output(this->m_shape, this->m_grad_flag);
    f32* restrict dy = output.storage_->data() + output.m_offset;

    for (size_t i = 0; i < num; ++i) dy[i] = std::pow(dx[i], value);

    return output;
}

MindTensor MindTensor::sum() const {
    MindTensor output({1}, this->m_grad_flag);

    const f32* restrict dx = this->storage_->data() + this->m_offset;
    f32* restrict dy = output.storage_->data();

    const size_t num = this->numel();
    avx2::vec8f acc = avx2::zero();
    size_t i = 0;
    for (; i + 8 <= num; i += 8) acc = avx2::add(acc, avx2::loadu(dx + i));

    f32 sum = avx2::hsum(acc);
    for (; i < num; ++i) sum += dx[i];

    dy[0] = sum;


    return output;
}

MindTensor MindTensor::sum(const i64 dim, const bool keep) const {
    CXM_ASSERT(dim >= 0 && dim < static_cast<int64_t>(this->m_shape.size()), "cortex::_fw::MindTensor::sum()", "Invalid dim");

    std::vector<int64_t> _shape;
    for (size_t i = 0; i < this->m_shape.size(); ++i) {
        if (i == static_cast<size_t>(dim)) {
            if (keep) _shape.push_back(1);
        } else {
            _shape.push_back(this->m_shape[i]);
        }
    }

    MindTensor output(_shape, requires_grad());

    const f32* restrict src = this->storage_->data() + this->m_offset;
    f32* restrict dst = output.storage_->data();

    const size_t dim_size = this->m_shape[dim];

    size_t outer = 1;
    for (size_t i = 0; i < static_cast<size_t>(dim); ++i) outer *= this->m_shape[i];

    size_t inner = 1;
    for (size_t i = dim + 1; i < this->m_shape.size(); ++i) inner *= this->m_shape[i];

    for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < inner; ++i) {
            avx2::vec8f acc = avx2::zero();
            size_t k = 0;

            for (; k + 8 <= dim_size; k += 8) {
                acc = avx2::add(acc, avx2::loadu(src + o * dim_size * inner + k * inner + i));
            }

            f32 sum = avx2::hsum(acc);

            for (; k < dim_size; ++k) sum += src[o * dim_size * inner + k * inner + i];

            dst[o * inner + i] = sum;
        }
    }

    return output;
}

MindTensor MindTensor::expand(const std::vector<i64>& shape) const {
    const auto old_dim = static_cast<i64>(this->m_shape.size());
    const auto new_dim = static_cast<i64>(shape.size());

    CXM_ASSERT(new_dim >= old_dim, "cortex::_fw::MindTensor::expand()", "Invalid shape");

    std::vector<int64_t> _strides(new_dim);

    const i64 _offset = new_dim - old_dim;

    for (i64 i = new_dim - 1; i >= 0; --i) {
        if (i - _offset >= 0) {
            const i64 oldSize = this->m_shape[i - _offset];

            if (const int64_t newSize = shape[i]; oldSize == newSize) {
                _strides[i] = this->m_strides[i - _offset];
            } else if (oldSize == 1) {
                _strides[i] = 0;
            } else {
                CXM_ASSERT(false, "cortex::_fw::MindTensor::expand", "Incompatible expand");
            }
        } else {
            _strides[i] = 0;
        }
    }

    MindTensor output(shape, this->m_grad_flag);
    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;
    output.m_strides = _strides;
    return output;
}

MindTensor MindTensor::transpose() {
    CXM_ASSERT(this->m_shape.size() == 2, "cortex::_fw::MindTensor::transpose()", "Tensor must be 2D to transpose.");

    MindTensor output({this->m_shape[1], this->m_shape[0]}, this->m_grad_flag);

    for (size_t i = 0; i < this->m_shape[0]; ++i) {
        for (size_t j = 0; j < this->m_shape[1]; ++j) {
            output.at(j * output.m_strides[0] + i * output.m_strides[1]) = this->at(i * this->m_strides[0] + j * this->m_strides[1]);
        }
    }
    
    output.m_offset = this->m_offset;
    return output;
}

MindTensor MindTensor::exp() const {
    MindTensor output(this->m_shape, this->m_grad_flag);

    const size_t num = this->numel();
    const f32* restrict dx = this->storage_->data() + this->m_offset;
    f32* restrict dy = output.storage_->data() + output.m_offset;

    size_t i = 0;
    for (; i + 7 < num; i += 8) {
        avx2::vec8f vx = avx2::loadu(dx + i);
        vx = avx2::exp(vx);
        avx2::storeu(dy + i, vx);
    }
    for (; i < num; ++i) dy[i] = std::exp(dx[i]);

    return output;
}

MindTensor MindTensor::log() const {
    MindTensor output(this->m_shape, this->m_grad_flag);

    const size_t num = this->numel();
    const f32* restrict dx = this->storage_->data() + this->m_offset;
    f32* restrict dy = output.storage_->data() + output.m_offset;

    size_t i = 0;
    for (; i + 7 < num; i += 8) {
        avx2::vec8f vx = avx2::loadu(dx + i);
        vx = avx2::log(vx);
        avx2::storeu(dy + i, vx);
    }
    for (; i < num; ++i) dy[i] = std::log(dx[i]);

    return output;
}

MindTensor MindTensor::abs() const {
    MindTensor output(this->m_shape, this->m_grad_flag);

    const size_t num = this->numel();
    const f32* restrict dx = this->storage_->data() + this->m_offset;
    f32* restrict dy = output.storage_->data() + output.m_offset;

    size_t i = 0;
    for (; i + 7 < num; i += 8) {
        avx2::vec8f vx = avx2::loadu(dx + i);
        vx = avx2::abs(vx);
        avx2::storeu(dy + i, vx);
    }
    for (; i < num; ++i) dy[i] = std::abs(dx[i]);

    return output;
}

MindTensor MindTensor::unsqueeze(i64 dim) const {
    std::vector<i64> _shape = this->m_shape;

    if (dim < 0) dim += static_cast<i64>(_shape.size());

    _shape.insert(_shape.begin() + dim, 1);

    MindTensor output(_shape, this->m_grad_flag);

    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;
    output.m_strides = compute_strides(output.m_shape);
    return output;
}

MindTensor MindTensor::squeeze(i64 dim) const {
    std::vector<i64> _shape = this->m_shape;

    if (dim < 0) dim += static_cast<i64>(_shape.size());

    _shape.erase(_shape.begin() + dim);

    MindTensor output(_shape, this->m_grad_flag);

    output.m_offset = this->m_offset;
    output.storage_ = this->storage_;

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
