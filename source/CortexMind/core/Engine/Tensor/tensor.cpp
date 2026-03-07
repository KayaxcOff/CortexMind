//
// Created by muham on 22.02.2026.
//

#include "CortexMind/core/Engine/Tensor/tensor.hpp"
#include <CortexMind/core/Engine/AVX2/matrix.hpp>
#include <CortexMind/core/Engine/AVX2/funcs.hpp>
#include <CortexMind/core/Engine/CL2/funcs.hpp>
#include <CortexMind/core/Engine/Memory/buffer.hpp>
#include <CortexMind/core/Graph/flow_ops.hpp>
#include <CortexMind/core/Tools/restrict.hpp>
#include <utility>
#include <functional>
#include <iomanip>
#include <random>

using namespace cortex::_fw;
using namespace cortex::_fw::sys;

MindTensor::MindTensor() : m_device(device::host), m_stride({}), m_shape({}), m_offset(0), m_require(false) {
    this->storage_ = nullptr;
    this->gradient_ = nullptr;
    this->flow_ = nullptr;
}

MindTensor::MindTensor(const std::vector<i64> &shape, const bool requires_grad) : m_shape(shape), m_offset(0), m_require(requires_grad) {
    CXM_ASSERT(is_valid_shape(this->m_shape), "cortex::_fw::MindTensor::MindTensor()", "Invalid shape.");
    this->m_device = device::host;

    this->m_stride = compute_strides(this->m_shape);

    this->storage_ = std::make_shared<TensorStorage>(this->numel());

    if (this->m_require) {
        this->gradient_ = std::make_unique<MindTensor>(this->m_shape);
        this->gradient_->zero();
    }
}

MindTensor::MindTensor(const std::initializer_list<i64> shape, const bool requires_grad) : MindTensor(std::vector(shape), requires_grad){
    this->m_device = device::host;
}

MindTensor::MindTensor(const std::vector<i64> &shape, const f32 *data, const bool requires_grad) : m_shape(shape), m_offset(0), m_require(requires_grad) {
    CXM_ASSERT(is_valid_shape(this->m_shape), "cortex::_fw::MindTensor::MindTensor()", "Invalid shape.");
    CXM_ASSERT(data != nullptr, "cortex::_fw::MindTensor::MindTensor()", "Dataset pointer can't be null.");

    this->m_device = device::host;

    this->m_stride = compute_strides(this->m_shape);

    this->storage_ = std::make_shared<TensorStorage>(this->numel());
    std::memcpy(this->storage_->data(), data, this->storage_->size() * sizeof(f32));

    if (this->m_require) {
        this->gradient_ = std::make_unique<MindTensor>(this->m_shape);
        this->gradient_->zero();
    }
}

MindTensor::MindTensor(const std::vector<i64> &shape, const device _device, const bool requires_grad) : m_device(_device), m_stride(compute_strides(shape)), m_shape(shape), m_offset(0), m_require(requires_grad) {
    CXM_ASSERT(is_valid_shape(this->m_shape), "cortex::_fw::MindTensor::MindTensor()", "Invalid shape.");
    this->storage_ = std::make_shared<TensorStorage>(compute_numel(this->m_shape), _device);

    if (_device == device::host) std::memset(this->storage_->data(), 0, compute_numel(this->m_shape) * sizeof(f32));

    if (this->m_require) {
        this->gradient_ = std::make_unique<MindTensor>(this->m_shape, _device, false);
        this->gradient_->zero();
    }
}

MindTensor::MindTensor(const MindTensor &other) : m_shape(other.m_shape), m_offset(other.m_offset), m_require(other.m_require) {
    this->m_device = other.m_device;

    this->storage_ = std::make_shared<TensorStorage>(*other.storage_);

    this->m_stride = other.m_stride;

    if (this->m_require) {
        this->gradient_ = std::make_unique<MindTensor>(*other.gradient_);
    }
}

MindTensor::MindTensor(MindTensor &&other) noexcept : storage_(std::move(other.storage_)), flow_(std::move(other.flow_)), m_device(other.m_device), m_stride(std::move(other.m_stride)), m_shape(std::move(other.m_shape)), m_offset(other.m_offset), m_require(other.m_require) {

    if (this->m_require) {
        this->gradient_ = std::move(other.gradient_);
    }

    other.m_offset  = 0;
    other.m_require = false;
}

MindTensor::~MindTensor() = default;

f32 &MindTensor::at(const std::vector<i64> &indices) {
    CXM_ASSERT(this->storage_ != nullptr, "cortex::_fw::MindTensor::at()", "Storage is null.");
    CXM_ASSERT(this->storage_->is_device(sys::device::host), "cortex::_fw::MindTensor::at()", "CPU access to the GPU tensor is not possible.");
    CXM_ASSERT(static_cast<i64>(indices.size()) == this->ndim(), "cortex::_fw::MindTensor::at()", "The index dimension does not match the tensor dimension.");

    for (i64 i = 0; i < this->ndim(); ++i) CXM_ASSERT(indices[i] >= 0 && indices[i] < this->m_shape[i], "cortex::_fw::MindTensor::at()", "Index delimited.");

    return this->storage_->data()[this->m_offset + compute_offset(indices, this->m_stride)];
}

const f32 &MindTensor::at(const std::vector<i64> &indices) const {
    CXM_ASSERT(this->storage_->isValid(), "cortex::_fw::MindTensor::at()", "Storage is null.");
    CXM_ASSERT(this->storage_->is_device(sys::device::host), "cortex::_fw::MindTensor::at()", "CPU access to the GPU tensor is not possible.");
    CXM_ASSERT(static_cast<i64>(indices.size()) == this->ndim(), "cortex::_fw::MindTensor::at()", "The index dimension does not match the tensor dimension.");

    for (i64 i = 0; i < this->ndim(); ++i) CXM_ASSERT(indices[i] >= 0 && indices[i] < this->m_shape[i], "cortex::_fw::MindTensor::at()", "Index delimited.");

    return this->storage_->data()[this->m_offset + compute_offset(indices, this->m_stride)];
}

f32 *MindTensor::get() {
    CXM_ASSERT(this->storage_->isValid(), "cortex::_fw::MindTensor::get()", "Storage is null.");
    CXM_ASSERT(this->storage_->is_device(sys::device::host), "cortex::_fw::MindTensor::get()", "CPU access to the GPU tensor is not possible.");
    return this->storage_->data() + this->m_offset;
}

f32 *MindTensor::get() const {
    CXM_ASSERT(this->storage_->isValid(), "cortex::_fw::MindTensor::get()", "Storage is null.");
    CXM_ASSERT(this->storage_->is_device(sys::device::host), "cortex::_fw::MindTensor::get()", "CPU access to the GPU tensor is not possible.");
    return this->storage_->data() + this->m_offset;
}

buffer *MindTensor::buffer() {
    CXM_ASSERT(this->storage_->isValid(), "cortex::_fw::MindTensor::get()", "Storage is null.");
    CXM_ASSERT(this->storage_->is_device(sys::device::cuda), "cortex::_fw::MindTensor::get()", "CPU access to the GPU tensor is not possible.");
    return this->storage_->buf();
}

buffer *MindTensor::buffer() const {
    CXM_ASSERT(this->storage_->isValid(), "cortex::_fw::MindTensor::get()", "Storage is null.");
    CXM_ASSERT(this->storage_->is_device(sys::device::cuda), "cortex::_fw::MindTensor::get()", "CPU access to the GPU tensor is not possible.");
    return this->storage_->buf();
}

std::vector<i64> MindTensor::shape() const noexcept {
    return this->m_shape;
}

i64 MindTensor::ndim() const noexcept {
    return static_cast<i64>(this->m_shape.size());
}

bool MindTensor::requires_grad() const noexcept {
    return this->m_require;
}

bool MindTensor::has_grad() const noexcept {
    return this->gradient_ != nullptr;
}

bool MindTensor::is_contiguous() const noexcept {
    return _fw::is_contiguous(this->m_shape, this->m_stride);
}

bool MindTensor::empty() const noexcept {
    return this->storage_ == nullptr || this->storage_->isEmpty();;
}

device MindTensor::devices() const noexcept {
    return this->m_device;
}

f32 MindTensor::mean() const {
    const size_t num = this->numel();
    CXM_ASSERT(num > 0, "cortex::_fw::MindTensor::mean()", "Tensor is empty.");

    if (this->m_device == device::host) {
        const f32* restrict px = this->get();
        avx2::vec8f acc = avx2::zero();
        size_t i = 0;
        for (; i + 8 <= num; i += 8)
            acc = avx2::add(acc, avx2::loadu(px + i));
        f32 sum = avx2::hsum(acc);
        for (; i < num; ++i) sum += px[i];
        return sum / static_cast<f32>(num);
    }
    if (this->m_device == device::cuda) {
        const f32 s = this->gpu_reduce("reduce_sum_partial", "reduce_sum_final", 0.0f);
        return s / static_cast<f32>(num);
    }
    CXM_ASSERT(false, "cortex::_fw::MindTensor::mean()", "Invalid device.");
    return 0.0f;
}

f32 MindTensor::variance() const {
    const size_t num  = this->numel();
    const f32    mu   = this->mean();
    CXM_ASSERT(num > 0, "cortex::_fw::MindTensor::variance()", "Tensor is empty.");

    if (this->m_device == device::host) {
        const f32* restrict px = this->get();
        const avx2::vec8f vmu = avx2::set1(mu);
        avx2::vec8f acc = avx2::zero();
        size_t i = 0;
        for (; i + 8 <= num; i += 8) {
            const avx2::vec8f diff = avx2::sub(avx2::loadu(px + i), vmu);
            acc = avx2::fma(diff, diff, acc);
        }
        f32 sum = avx2::hsum(acc);
        for (; i < num; ++i) sum += (px[i] - mu) * (px[i] - mu);
        return sum / static_cast<f32>(num);
    }
    if (this->m_device == device::cuda) {

        const i64    n        = static_cast<i64>(num);
        const size_t n_groups = cl2::align_to(n, cl2::BLOCK_SIZE) / cl2::BLOCK_SIZE;

        sys::buffer partial(n_groups);
        sys::buffer result(1);

        cl2::registry::get().reduce().run(
            "reduce_var_partial",
            cl::NDRange(cl2::align_to(n, cl2::BLOCK_SIZE)),
            cl::NDRange(cl2::BLOCK_SIZE),
            this->storage_->buf()->handle(), mu, partial,
            cl::Local(cl2::BLOCK_SIZE * sizeof(cl_float)),
            static_cast<cl_int>(n)
        );
        cl2::registry::get().reduce().run(
            "reduce_sum_final",
            cl::NDRange(cl2::BLOCK_SIZE),
            cl::NDRange(cl2::BLOCK_SIZE),
            partial, result,
            cl::Local(cl2::BLOCK_SIZE * sizeof(cl_float)),
            static_cast<cl_int>(n_groups)
        );
        f32 out;
        result.download(&out, 1);
        return out / static_cast<f32>(num);
    }
    CXM_ASSERT(false, "cortex::_fw::MindTensor::variance()", "Invalid device.");
    return 0.0f;
}

f32 MindTensor::std_dev() const {
    return std::sqrt(this->variance());
}

f32 MindTensor::max() const {
    const size_t num = this->numel();
    CXM_ASSERT(num > 0, "cortex::_fw::MindTensor::max()", "Tensor is empty.");

    if (this->m_device == device::host) {
        const f32* restrict px = this->get();
        avx2::vec8f acc = avx2::set1(std::numeric_limits<f32>::lowest());
        size_t i = 0;
        for (; i + 8 <= num; i += 8)
            acc = avx2::max(acc, avx2::loadu(px + i));
        f32 result = avx2::hmax(acc);
        for (; i < num; ++i) result = std::max(result, px[i]);
        return result;
    }
    if (this->m_device == device::cuda)
        return this->gpu_reduce("reduce_max_partial", "reduce_max_final", 0.0f);
    CXM_ASSERT(false, "cortex::_fw::MindTensor::max()", "Invalid device.");
    return 0.0f;
}

f32 MindTensor::min() const {
    const size_t num = this->numel();
    CXM_ASSERT(num > 0, "cortex::_fw::MindTensor::min()", "Tensor is empty.");

    if (this->m_device == device::host) {
        const f32* restrict px = this->get();
        avx2::vec8f acc = avx2::set1(std::numeric_limits<f32>::max());
        size_t i = 0;
        for (; i + 8 <= num; i += 8)
            acc = avx2::min(acc, avx2::loadu(px + i));
        f32 result = avx2::hmin(acc);
        for (; i < num; ++i) result = std::min(result, px[i]);
        return result;
    }
    if (this->m_device == device::cuda)
        return gpu_reduce("reduce_min_partial", "reduce_min_final", 0.0f);
    CXM_ASSERT(false, "cortex::_fw::MindTensor::min()", "Invalid device.");
    return 0.0f;
}

f32 MindTensor::norm() const {
    const size_t num = this->numel();
    CXM_ASSERT(num > 0, "cortex::_fw::MindTensor::norm()", "Tensor is empty.");

    if (this->m_device == device::host) {
        const f32* restrict px = this->get();
        avx2::vec8f acc = avx2::zero();
        size_t i = 0;
        for (; i + 8 <= num; i += 8) {
            const avx2::vec8f v = avx2::loadu(px + i);
            acc = avx2::fma(v, v, acc);
        }
        f32 result = avx2::hsum(acc);
        for (; i < num; ++i) result += px[i] * px[i];
        return std::sqrt(result);
    }
    if (this->m_device == device::cuda) {

    }
    CXM_ASSERT(false, "cortex::_fw::MindTensor::norm()", "Invalid device.");
    return 0.0f;
}

f32 MindTensor::sum_all() const {
    f32 output = 0.0f;
    for (size_t i = 0; i < this->numel(); ++i) {
        output += this->get()[i];
    }
    return output;
}

size_t MindTensor::numel() const noexcept{
    size_t output = 1;
    for (const auto& item : this->m_shape) output *= item;
    return output;
}

void MindTensor::backward() const {
    CXM_ASSERT(this->m_require, "cortex::_fw::MindTensor::backward()", "Grad flag is false so you can't use gradient");
    CXM_ASSERT(this->gradient_.get(), "cortex::_fw::MindTensor::backward()", "Gradient is null");

    if (this->numel() == 1) this->gradient_->ones();

    if (this->flow_) this->flow_->backward(*this->gradient_);
}

void MindTensor::keep_backward(MindTensor *_grad) const {
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

void MindTensor::print_shape() const {
    std::cout << "(";
    for (i64 i = 0; i < this->m_shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << this->m_shape[i];
    }
    std::cout << ")" << std::endl;
}

void MindTensor::print_stride() const {
    std::cout << "(";
    for (i64 i = 0; i < this->m_shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << this->m_stride[i];
    }
    std::cout << ")" << std::endl;
}

void MindTensor::uniform_rand(f32 min, f32 max) {
    if (min > max) std::swap(min, max);

    if (this->m_device == device::host) {
        thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution dist(min, max);

        const size_t num = this->numel();
        f32* restrict px = this->get();

        for (size_t i = 0; i < num; ++i) px[i] = dist(gen);
    }
    if (this->m_device == device::cuda) {}
}

void MindTensor::zero() const {
    std::memset(this->storage_->data() + this->m_offset, 0, this->numel() * sizeof(f32));
}

void MindTensor::ones() const {
    const size_t num = this->numel();
    f32* restrict px = this->storage_->data() + this->m_offset;
    const avx2::vec8f v1 = avx2::set1(1.0f);
    size_t i = 0;
    for (; i + 8 <= num; i += 8) avx2::storeu(px + i, v1);
    for (; i < num; ++i) px[i] = 1.0f;
}

void MindTensor::fill(const f32 val) const {
    const size_t num = this->numel();
    f32* restrict px = this->storage_->data() + this->m_offset;
    const avx2::vec8f vval = avx2::set1(val);
    size_t i = 0;
    for (; i + 8 <= num; i += 8) avx2::storeu(px + i, vval);
    for (; i < num; ++i) px[i] = val;
}

void MindTensor::require_grad(const bool _require_grad) {
    this->m_require = _require_grad;

    if (this->m_require && !this->gradient_) {
        this->gradient_ = std::make_unique<MindTensor>(this->m_shape);
        this->gradient_->zero();
    }
}

void MindTensor::set_grad(std::unique_ptr<MindTensor> _grad) {
    CXM_ASSERT(this->m_require, "cortex::_fw::MindTensor::set_grad()", "Gradient flag is false so you can't use gradient");
    this->gradient_ = std::move(_grad);
}

void MindTensor::set_grad(const MindTensor &grad) {
    CXM_ASSERT(this->m_require, "cortex::_fw::MindTensor::set_grad()", "Gradient flag is false so you can't use gradient");
    this->gradient_ = std::make_unique<MindTensor>(grad);
}

void MindTensor::zero_grad() const {
    CXM_ASSERT(this->m_require, "cortex::_fw::MindTensor::zero_grad()", "Gradient flag is false so you can't use gradient");
    this->gradient_->zero();
}

void MindTensor::set_flow(std::shared_ptr<meta::GradientFlow> _flow) {
    this->flow_ = std::move(_flow);
}

void MindTensor::clear_flow() {
    this->flow_ = nullptr;
}

MindTensor MindTensor::to(const device _device) {
    if (this->m_device == _device) return *this;

    this->storage_ = std::make_shared<TensorStorage>(this->storage_->to(_device));
    this->m_device = _device;
    return *this;
}

MindTensor MindTensor::flatten() const {
    i64 batch = this->m_shape[0];

    i64 feat = 1;
    for (size_t i = 1; i < this->m_shape.size(); ++i) feat *= this->m_shape[i];

    if (!this->is_contiguous()) {
        CXM_ASSERT(this->m_device == device::host, "cortex::_fw::MindTensor::flatten()", "Non-contiguous GPU tensor flatten not yet supported.");
        const MindTensor output = this->clone();
        return output.flatten();
    }

    MindTensor output({batch, feat}, this->m_device, this->m_require);
    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;
    return output;
}

MindTensor MindTensor::matmul(const MindTensor &other) {
    MindTensor a = this->is_contiguous()  ? *this  : this->clone();
    MindTensor b = other.is_contiguous()  ? other  : other.clone();

    CXM_ASSERT(this->m_stride.size() == 2 && other.m_stride.size() == 2, "cortex::_fw::MindTensor::matmul()", "Both tensors must be 2D for matrix multiplication");
    CXM_ASSERT(this->m_shape[1] == other.m_shape[0], "cortex::_fw::MindTensor::matmul()", "Inner dimensions must match for matrix multiplication");
    CXM_ASSERT(this->m_device == other.m_device, "cortex::_fw::MindTensor::matmul()", "Both tensors must be on the same device.");

    MindTensor output({this->m_shape[0], other.m_shape[1]}, this->m_device, this->m_require || other.m_require);

    if (this->m_device == device::host) {
        avx2::matrix_t::matmul(this->get(), other.get(), output.get(), static_cast<size_t>(this->m_shape[0]), static_cast<size_t>(this->m_shape[1]), static_cast<size_t>(other.m_shape[1]));
    }
    if (this->m_device == device::cuda) {
        cl2::matmul(*this->buffer(), *other.buffer(), *output.buffer(), static_cast<size_t>(this->m_shape[0]), static_cast<size_t>(this->m_shape[1]), static_cast<size_t>(other.m_shape[1]));
    }

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::matmul>(this, const_cast<MindTensor*>(&other));
    }

    return output;
}

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

    MindTensor output(_shape, this->m_require);

    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;
    output.m_stride = _stride;
    output.m_device = this->m_device;
    return output;
}

MindTensor MindTensor::clone() const {
    MindTensor output(this->m_shape, this->m_device, this->m_require);

    if (this->m_device == device::host) {
        f32* restrict dst = output.get();
        const size_t num  = this->numel();

        if (this->is_contiguous()) {
            std::memcpy(dst, this->get(), num * sizeof(f32));
        } else {
            std::vector<i64> idx(this->ndim(), 0);
            for (size_t flat = 0; flat < num; ++flat) {
                const i64 src_offset = this->m_offset + compute_offset(idx, this->m_stride);
                dst[flat] = this->storage_->data()[src_offset];

                for (i64 d = ndim() - 1; d >= 0; --d) {
                    if (++idx[d] < this->m_shape[d]) break;
                    idx[d] = 0;
                }
            }
        }
    }

    if (this->gradient_) output.gradient_ = std::make_unique<MindTensor>(this->gradient_->clone());

    return output;
}

MindTensor MindTensor::detach() const {
    MindTensor output(this->m_shape, false);

    output.m_device  = this->m_device;
    output.m_offset  = this->m_offset;
    output.m_stride  = this->m_stride;
    output.storage_  = this->storage_;

    return output;
}

MindTensor MindTensor::reshape(const std::vector<i64> &shape) const {
    CXM_ASSERT(this->is_contiguous(), "cortex::_fw::MindTensor::reshape()", "Tensor must be contiguous");

    CXM_ASSERT(compute_numel(shape) == this->numel(), "cortex::_fw::MindTensor::reshape()", "Total elements must be same");

    MindTensor output(shape, this->m_require);

    output.storage_ = this->storage_;
    output.m_offset = m_offset;
    output.m_stride = compute_strides(output.m_shape);
    output.m_device = this->m_device;
    return output;
}

MindTensor MindTensor::sqrt() const {
    MindTensor output(this->m_shape, this->m_device, this->m_require);

    if (this->m_device == device::host) {
        const f32* restrict px = this->get();
        f32*       restrict py = output.get();
        const size_t num = this->numel();

        size_t i = 0;
        for (; i + 8 <= num; i += 8) {
            avx2::vec8f vx = avx2::loadu(px + i);
            avx2::storeu(py + i, avx2::sqrt(vx));
        }
        for (; i < num; ++i) py[i] = std::sqrt(px[i]);

    } else if (this->m_device == device::cuda) {
        cl2::sqrt(*this->buffer(), *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::sqrt()", "Invalid device.");
    }

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::sqrt>(const_cast<MindTensor*>(this));
    }

    return output;
}

MindTensor MindTensor::pow(const f32 value) const {
    MindTensor output(this->m_shape, this->m_device, this->m_require);

    if (this->m_device == device::host) {
        const f32* restrict px = this->get();
        f32*       restrict py = output.get();
        const size_t num = this->numel();

        size_t i = 0;
        for (; i + 8 <= num; i += 8) {
            avx2::vec8f vx = avx2::loadu(px + i);
            avx2::storeu(py + i, avx2::pow(vx, avx2::set1(value)));
        }
        for (; i < num; ++i) py[i] = std::pow(px[i], value);

    } else if (this->m_device == device::cuda) {
        cl2::pow(*this->buffer(), value, *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::pow()", "Invalid device.");
    }

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::pow>(const_cast<MindTensor*>(this), value);
    }

    return output;
}

MindTensor MindTensor::sum() const {
    MindTensor output({1}, this->m_device, this->m_require);

    if (this->m_device == device::host) {
        const f32* restrict px = this->get();
        f32* restrict py = output.get();
        const size_t num = this->numel();

        avx2::vec8f vx = avx2::zero();
        size_t i = 0;
        for (; i + 8 <= num; i += 8) vx = avx2::add(vx, avx2::loadu(px + i));
        f32 sum = avx2::hsum(vx);
        for (; i < num; ++i) sum += px[i];

        py[0] = sum;
    } else if (this->m_device == device::cuda) {
        const f32 result = this->gpu_reduce("reduce_sum_partial", "reduce_sum_final", 0.0f);
        output.buffer()->upload(&result, 1);
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::sum()", "Invalid device.");
    }

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::sum>(const_cast<MindTensor*>(this));
    }

    return output;
}

MindTensor MindTensor::sum(const i64 dim, const bool keep) const {
    CXM_ASSERT(dim >= 0 && dim < this->ndim(), "cortex::_fw::MindTensor::sum()", "Dimension out of range.");
    CXM_ASSERT(this->m_device == device::host, "cortex::_fw::MindTensor::sum()", "GPU dim-wise sum not yet implemented.");

    std::vector<i64> out_shape;
    out_shape.reserve(this->m_shape.size());

    for (i64 i = 0; i < ndim(); ++i) {
        if (i == dim) {
            if (keep) out_shape.push_back(1);
        } else {
            out_shape.push_back(this->m_shape[i]);
        }
    }

    if (out_shape.empty()) out_shape.push_back(1);

    MindTensor output(out_shape, this->m_device, this->m_require);
    output.zero();

    const i64    dim_size = this->m_shape[dim];

    const size_t out_numel = output.numel();
    const f32* restrict src = this->get();
    f32* restrict       dst = output.get();

    for (size_t out_flat = 0; out_flat < out_numel; ++out_flat) {

        std::vector<i64> out_idx(out_shape.size());
        {
            size_t tmp = out_flat;
            for (i64 i = static_cast<i64>(out_shape.size()) - 1; i >= 0; --i) {
                out_idx[i] = static_cast<i64>(tmp) % out_shape[i];
                tmp /= static_cast<size_t>(out_shape[i]);
            }
        }

        std::vector<i64> in_idx(this->m_shape.size());
        {
            i64 out_i = 0;
            for (i64 in_i = 0; in_i < ndim(); ++in_i) {
                if (in_i == dim) {
                    in_idx[in_i] = 0;
                } else {
                    if (keep) {
                        in_idx[in_i] = out_idx[out_i];
                    } else {
                        in_idx[in_i] = out_idx[out_i];
                    }
                    ++out_i;
                }
            }
        }

        f32 acc = 0.0f;
        for (i64 j = 0; j < dim_size; ++j) {
            in_idx[dim] = j;
            const i64 in_flat = this->m_offset + compute_offset(in_idx, this->m_stride);
            acc += src[in_flat];
        }

        dst[out_flat] = acc;
    }

    return output;
}

MindTensor MindTensor::expand(const std::vector<i64>& shape) const {
    const auto old_dim = static_cast<i64>(this->m_shape.size());
    const auto new_dim = static_cast<i64>(shape.size());

    CXM_ASSERT(new_dim >= old_dim, "cortex::_fw::MindTensor::expand()", "Invalid shape.");
    CXM_ASSERT(is_broadcastable(this->m_shape, shape), "cortex::_fw::MindTensor::expand()", "Shapes are not broadcastable.");

    std::vector<i64> _strides(new_dim);
    const i64 _offset = new_dim - old_dim;

    for (i64 i = new_dim - 1; i >= 0; --i) {
        if (i - _offset >= 0) {
            const i64 old_size = this->m_shape[i - _offset];
            const i64 new_size = shape[i];

            if (old_size == new_size) {
                _strides[i] = this->m_stride[i - _offset];
            } else if (old_size == 1) {
                _strides[i] = 0;
            } else {
                CXM_ASSERT(false, "cortex::_fw::MindTensor::expand()", "Incompatible dimensions.");
            }
        } else {
            _strides[i] = 0;
        }
    }

    MindTensor output(shape, this->m_device, this->m_require);
    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;
    output.m_stride = _strides;

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::expand>(const_cast<MindTensor*>(this), this->m_shape);
    }

    return output;
}

MindTensor MindTensor::repeat(const i64 times, const i64 dim) {
    std::vector<i64> new_shape = this->m_shape;
    new_shape[dim] *= times;

    MindTensor output(new_shape, this->m_device, this->m_require);
    const auto row_size = static_cast<size_t>(this->m_shape[1]);

    for (i64 i = 0; i < times; ++i)
        std::memcpy(output.get() + i * row_size,
                    this->get(),
                    row_size * sizeof(f32));

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::repeat>(this, times, dim);
    }
    return output;
}

MindTensor MindTensor::transpose() {
    CXM_ASSERT(this->ndim() == 2, "cortex::_fw::MindTensor::transpose()", "Tensor must be 2D.");

    MindTensor output({this->m_shape[1], this->m_shape[0]}, this->m_device, this->m_require);
    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;
    output.m_stride = {this->m_stride[1], this->m_stride[0]};

    return output;
}

MindTensor MindTensor::exp() const {
    MindTensor output(this->m_shape, this->m_device, this->m_require);

    if (this->m_device == device::host) {
        const size_t num = this->numel();
        const f32* restrict px = this->get();
        f32* restrict py = output.get();

        size_t i = 0;
        for (; i + 7 < num; i += 8) {
            avx2::storeu(py + i, avx2::exp(avx2::loadu(px + i)));
        }
        for (; i < num; ++i) py[i] = std::exp(px[i]);
    } else if (this->m_device == device::cuda) {
        cl2::exp(*this->buffer(), *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::sum()", "Invalid device.");
    }

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::exp>(const_cast<MindTensor*>(this));
    }

    return output;
}

MindTensor MindTensor::log() const {
    MindTensor output(this->m_shape, this->m_device, this->m_require);

    if (this->m_device == device::host) {
        const size_t num = this->numel();
        const f32* restrict px = this->get();
        f32* restrict py = output.get();

        size_t i = 0;
        for (; i + 7 < num; i += 8) {
            avx2::storeu(py + i, avx2::log(avx2::loadu(px + i)));
        }
        for (; i < num; ++i) py[i] = std::log(px[i]);
    } else if (this->m_device == device::cuda) {
        cl2::log(*this->buffer(), *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::log()", "Invalid device.");
    }

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::log>(const_cast<MindTensor*>(this));
    }

    return output;
}

MindTensor MindTensor::abs() const {
    MindTensor output(this->m_shape, this->m_device, this->m_require);

    if (this->m_device == device::host) {
        const size_t num = this->numel();
        const f32* restrict px = this->get();
        f32* restrict py = output.get();

        size_t i = 0;
        for (; i + 7 < num; i += 8) {
            avx2::storeu(py + i, avx2::abs(avx2::loadu(px + i)));
        }
        for (; i < num; ++i) py[i] = std::abs(px[i]);
    } else if (this->m_device == device::cuda) {
        cl2::abs(*this->buffer(), *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::abs()", "Invalid device.");
    }

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::abs>(const_cast<MindTensor*>(this));
    }

    return output;
}

MindTensor MindTensor::relu() {
    MindTensor output(this->m_shape, this->m_device, this->m_require);

    if (this->m_device == device::host) {
        const f32* restrict px = this->get();
        f32* restrict py = output.get();
        const size_t num = this->numel();

        size_t i = 0;
        for (; i + 8 <= num; i += 8) {
            avx2::store(py + i, avx2::relu(avx2::loadu(px + i)));
        }
        for (; i < num; ++i) py[i] = std::max(px[i], 0.0f);
    } else if (this->m_device == device::cuda) {
        cl2::relu(*this->buffer(), *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::relu()", "Invalid device");
    }

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::relu>(this);
    }

    return output;
}

MindTensor MindTensor::tanh() const {
    MindTensor output(this->m_shape, this->m_device, this->m_require);

    if (this->m_device == device::host) {
        const size_t num = this->numel();
        const f32* restrict px = this->get();
        f32* restrict py = output.get();

        size_t i = 0;
        for (; i + 8 <= num; i += 8) {
            avx2::storeu(py + i, avx2::tanh(avx2::loadu(px + i)));
        }
        for (; i < num; ++i) py[i] = std::tanh(px[i]);
    } else if (this->m_device == device::cuda) {
        cl2::tanh(*this->buffer(), *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::tanh()", "Invalid tensor");
    }

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::tanh>(const_cast<MindTensor*>(this));
    }

    return output;
}

MindTensor MindTensor::sigmoid() {
    MindTensor output(this->m_shape, this->m_device, this->m_require);

    if (this->m_device == device::host) {
        const size_t num = this->numel();
        const f32* restrict px = this->get();
        f32* restrict py = output.get();

        size_t i = 0;
        for (; i + 8 <= num; i += 8) {
            // sigmoid(x) = 1 / (1 + exp(-x))
            const avx2::vec8f vx  = avx2::loadu(px + i);
            const avx2::vec8f neg = avx2::mul(vx, avx2::set1(-1.0f));
            const avx2::vec8f ex  = avx2::exp(neg);
            const avx2::vec8f one = avx2::set1(1.0f);
            avx2::storeu(py + i, avx2::div(one, avx2::add(one, ex)));
        }
        for (; i < num; ++i)
            py[i] = 1.0f / (1.0f + std::exp(-px[i]));

    } else if (this->m_device == device::cuda) {
        cl2::sigmoid(*this->buffer(), *output.buffer());
    } else {
        CXM_ASSERT(false, "cortex::_fw::MindTensor::sigmoid()", "Invalid device.");
    }

    return output;
}

MindTensor MindTensor::unsqueeze(i64 dim) const {
    std::vector<i64> _shape = this->m_shape;

    if (dim < 0) dim += static_cast<i64>(_shape.size());

    _shape.insert(_shape.begin() + dim, 1);

    MindTensor output(_shape, this->m_device, this->m_require);

    output.storage_ = this->storage_;
    output.m_offset = this->m_offset;
    output.m_stride = compute_strides(output.m_shape);
    return output;
}

MindTensor MindTensor::squeeze(i64 dim) const {
    std::vector<i64> _shape = this->m_shape;

    if (dim < 0) dim += static_cast<i64>(_shape.size());

    _shape.erase(_shape.begin() + dim);

    MindTensor output(_shape, this->m_device, this->m_require);

    output.m_offset = this->m_offset;
    output.storage_ = this->storage_;

    return output;
}

MindTensor MindTensor::eq(const MindTensor &other) const {
    return elementwise_cmp(*this, other,
        "cortex::_fw::MindTensor::eq()",
        avx2::cmp::eq,
        [](const f32 a, const f32 b) { return a == b; }, cl2::eq);
}

MindTensor MindTensor::ne(const MindTensor &other) const {
    return elementwise_cmp(*this, other,
        "cortex::_fw::MindTensor::ne()",
        [](const avx2::vec8f& x, const avx2::vec8f& y) {
            const avx2::vec8f eq_mask = avx2::cmp::eq(x, y);
            const avx2::vec8f all_ones = avx2::set1(-1.0f); // 0xFFFFFFFF
            return avx2::xor_(eq_mask, all_ones);
        },
        [](const f32 a, const f32 b) { return a != b; },
        cl2::ne);
}

MindTensor MindTensor::gt(const MindTensor &other) const {
    return elementwise_cmp(*this, other,
        "cortex::_fw::MindTensor::gt()",
        avx2::cmp::gt,
        [](const f32 a, const f32 b) { return a > b; },
        cl2::gt);
}

MindTensor MindTensor::lt(const MindTensor &other) const {
    return elementwise_cmp(*this, other,
        "cortex::_fw::MindTensor::lt()",
        avx2::cmp::lt,
        [](const f32 a, const f32 b) { return a < b; },
        cl2::lt);
}

MindTensor MindTensor::slice(const i64 start, const i64 end) const {
    CXM_ASSERT(this->ndim() == 2, "cortex::_fw::MindTensor::slice()", "Only 2D tensors supported.");
    CXM_ASSERT(start >= 0 && end <= this->m_shape[0] && start < end,
               "cortex::_fw::MindTensor::slice()", "Invalid slice range.");

    const i64 rows = end - start;
    const i64 cols = this->m_shape[1];

    MindTensor output({rows, cols}, this->m_device, this->m_require);
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

f32 MindTensor::gpu_reduce(const std::string &partial_kernel, const std::string &final_kernel, cl_float identity) const {
    const i64    n        = compute_numel(this->m_shape);
    const size_t n_groups = cl2::align_to(n, cl2::BLOCK_SIZE) / cl2::BLOCK_SIZE;

    sys::buffer partial(n_groups);
    sys::buffer result(1);

    cl2::registry::get().reduce().run(
        partial_kernel,
        cl::NDRange(cl2::align_to(n, cl2::BLOCK_SIZE)),
        cl::NDRange(cl2::BLOCK_SIZE),
        this->storage_->buf()->handle(), partial,
        cl::Local(cl2::BLOCK_SIZE * sizeof(cl_float)),
        static_cast<cl_int>(n)
    );

    cl2::registry::get().reduce().run(
        final_kernel,
        ::cl::NDRange(cl2::BLOCK_SIZE),
        ::cl::NDRange(cl2::BLOCK_SIZE),
        partial, result,
        cl::Local(cl2::BLOCK_SIZE * sizeof(cl_float)),
        static_cast<cl_int>(n_groups)
    );

    f32 out;
    result.download(&out, 1);
    return out;
}

MindTensor MindTensor::elementwise_cmp(const MindTensor &a, const MindTensor &b, const char *name, avx2::vec8f (*avx2_op)(const avx2::vec8f &, const avx2::vec8f &), bool (*scalar_op)(f32, f32), void (*cl_op)(const sys::buffer &, const sys::buffer &, sys::buffer &)) {
    CXM_ASSERT(a.m_device == b.m_device, name, "Tensors must be on same device.");
    CXM_ASSERT(a.m_shape == b.m_shape,   name, "Shapes don't match.");

    MindTensor output(a.m_shape, a.m_device, false);

    if (a.m_device == device::host) {
        const f32* restrict px = a.get();
        const f32* restrict py = b.get();
        f32*       restrict pz = output.get();
        const size_t num = a.numel();

        size_t i = 0;
        for (; i + 8 <= num; i += 8) {
            const avx2::vec8f vx   = avx2::loadu(px + i);
            const avx2::vec8f vy   = avx2::loadu(py + i);
            const avx2::vec8f mask = avx2_op(vx, vy);
            // mask: 0xFFFFFFFF → 1.0f, 0x00000000 → 0.0f
            const avx2::vec8f one  = avx2::set1(1.0f);
            const avx2::vec8f zero = avx2::zero();
            avx2::storeu(pz + i, avx2::blendv(zero, one, mask));
        }
        for (; i < num; ++i)
            pz[i] = scalar_op(px[i], py[i]) ? 1.0f : 0.0f;

    } else if (a.m_device == sys::device::cuda) {
        cl_op(*a.buffer(), *b.buffer(), *output.buffer());
    } else {
        CXM_ASSERT(false, name, "Invalid device.");
    }
    return output;
}
