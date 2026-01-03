//
// Created by muham on 29.12.2025.
//

#include "core/Engine/Tensor/tensor.hpp"
#include <core/Engine/AVX/ops.hpp>
#include <core/Engine/AVX/matrix.hpp>
#include <core/Tools/debug.hpp>
#include <iomanip>
#include <iostream>
#include <random>

using namespace cortex::_fw;

std::vector<int64_t> MindTensor::compute_strides(const std::vector<int64_t> &shape) noexcept {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(strides.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

int64_t MindTensor::compute_offset(const std::vector<int64_t> &indices) const noexcept {
    int64_t offset = 0;
    for (int64_t i = 0; i < indices.size(); ++i) {
        err::IsAnError(indices[i] >= 0 && indices[i] < this->m_shape[i], "cortex::_fw::MindTensor::compute_offset()", "Out of bounds access detected");
        offset += indices[i] * this->m_strides[i];
    }
    return offset + this->m_offset;
}

bool MindTensor::is_contiguous() const noexcept {
    return this->m_strides == compute_strides(this->m_shape);
}

MindTensor::MindTensor(const bool requires_grad) : m_shape({}), m_strides({}) {
    if (requires_grad) {
        this->m_grad = std::make_shared<MindTensor>();
        this->m_flow = std::make_shared<meta::AutoDiff<MindTensor>>();
        this->m_flow->requires_grad = true;
    }
}

MindTensor::MindTensor(const std::vector<int64_t> &shape, const bool requires_grad) : m_shape(shape), m_strides(compute_strides(m_shape)) {
    size_t total = this->m_shape.empty() ? 0 : (this->m_strides[0] * this->m_shape[0]);

    this->m_stor = std::make_shared<TensorStorage>(total);

    if (requires_grad) {
        this->m_grad = std::make_shared<MindTensor>();
        this->m_flow = std::make_shared<meta::AutoDiff<MindTensor>>();
        this->m_flow->requires_grad = true;
    }
}

MindTensor::MindTensor(const std::initializer_list<int64_t> shape, const bool requires_grad) : MindTensor(std::vector(shape), requires_grad) {}

MindTensor::MindTensor(const std::vector<int64_t>& shape, const float* data, const bool requires_grad) : MindTensor(shape, requires_grad) {
    size_t total = 1;
    for (const auto& dim : this->m_shape) {
        total *= dim;
    }
    err::IsAnError(data != nullptr, "cortex::_fw::MindTensor::MindTensor()", "Input data pointer is null.");
    std::memcpy(this->m_stor->ptr(), data, total * sizeof(float));

    if (requires_grad) {
        this->m_grad = std::make_shared<MindTensor>();
        this->m_flow = std::make_shared<meta::AutoDiff<MindTensor>>();
        this->m_flow->requires_grad = true;
    }
}

float &MindTensor::at(const std::vector<int64_t> &indices) noexcept {
    const int64_t offset = this->compute_offset(indices);
    return this->m_stor->ptr()[offset];
}

const float &MindTensor::at(const std::vector<int64_t> &indices) const noexcept {
    const int64_t offset = this->compute_offset(indices);
    return this->m_stor->ptr()[offset];
}

float* MindTensor::data() noexcept {
    return this->m_stor->ptr() + this->m_offset;
}

const float* MindTensor::data() const noexcept {
    return this->m_stor->ptr() + this->m_offset;
}

const std::vector<int64_t>& MindTensor::shape() const noexcept {
    return this->m_shape;
}

const std::vector<int64_t>& MindTensor::strides() const noexcept {
    return this->m_strides;
}

size_t MindTensor::total() const noexcept {
    size_t total = 1;
    for (const auto& item : this->m_shape) total *= item;
    return total;
}

std::size_t MindTensor::size() const noexcept {
    return this->m_stor->size();
}

bool MindTensor::empty() const noexcept {
    return this->m_stor->empty();
}

bool MindTensor::requires_grad() const noexcept {
    return m_flow ? m_flow->requires_grad : false;
}

float MindTensor::mean() const {
    const auto total = this->total();
    err::IsAnError(total > 0, "cortex::_fw::MindTensor::mean()", "Cannot compute mean of an empty tensor");
    const float* it = this->m_stor->ptr() + this->m_offset;

    avx2::vec8f vt = avx2::zero();

    size_t i = 0;
    for (; i + 8 < total; i += 8) {
        const avx2::vec8f vd = avx2::load_u(it + i);
        vt = avx2::add(vt, vd);
    }
    float sum = 0;
    for (; i < total; ++i) sum += it[i];

    const float hs = avx2::hadd(vt);

    return (hs + sum) / static_cast<float>(total);
}

void MindTensor::backward() const {
    err::IsAnError(this->requires_grad(), "cortex::_fw::MindTensor::backward()", "Tensor does not require grad");
}

void MindTensor::print() const noexcept {
    if (m_shape.empty()) {
        std::cout << "Tensor is empty\n";
        return;
    }

    constexpr int indent_step = 2;

    std::function<void(int64_t, int64_t, int)> PrintRecursive;
    PrintRecursive = [&](const int64_t dim, const int64_t offset, const int indent) {
        if (dim == static_cast<int64_t>(this->m_shape.size()) - 1) {
            std::cout << "[";
            for (int64_t i = 0; i < this->m_shape[dim]; ++i) {
                std::cout << std::fixed << std::setprecision(4) << this->m_stor->ptr()[offset + i * this->m_strides[dim]];
                if (i + 1 != this->m_shape[dim]) std::cout << ", ";
            }
            std::cout << "]";
            return;
        }
        std::cout << "[";
        for (int64_t i = 0; i < this->m_shape[dim]; ++i) {
            if (i == 0) {
                if (dim > 0) std::cout << "\n" << std::string(indent + indent_step, ' ');
            } else {
                std::cout << ",\n" << std::string(indent + indent_step, ' ');
            }

            PrintRecursive(dim + 1, offset + i * this->m_strides[dim], indent + indent_step);
        }

        if (dim > 0) {
            std::cout << "\n" << std::string(indent, ' ');
        }
        std::cout << "]";
    };

    PrintRecursive(0, m_offset, 0);
    std::cout << std::endl;
}

void MindTensor::uniform_rand(const float lower, const float upper) const noexcept {
    thread_local std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution dist{lower, upper};

    const size_t total = this->total();
    float* it = this->m_stor->ptr() + this->m_offset;

    for (size_t i = 0; i < total; ++i) it[i] = dist(gen);
}

void MindTensor::zero() const noexcept {
    const size_t total = this->total();

    float* it = this->m_stor->ptr() + this->m_offset;

    const avx2::vec8f vt = avx2::zero();

    size_t i = 0;
    for (; i + 8 < total; i += 8) avx2::store_u(it + i, vt);
    for (; i < total; ++i) it[i] = 0;
}

void MindTensor::ones() const noexcept {
    const size_t total = this->total();

    float* it = this->m_stor->ptr() + this->m_offset;

    const avx2::vec8f vt = avx2::broadcast(1.0);

    size_t i = 0;
    for (; i + 8 < total; i += 8) avx2::store_u(it + i, vt);
    for (; i < total; ++i) it[i] = 0;
}

void MindTensor::fill(const float value) const noexcept {
    const size_t total = this->total();

    float* it = this->m_stor->ptr() + this->m_offset;
    const avx2::vec8f vt = avx2::broadcast(value);

    size_t i = 0;
    for (; i + 8 < total; i += 8) avx2::store_u(it + i, vt);
    for (; i < total; ++i) it[i] = value;
}

void MindTensor::allocate() noexcept {
    if (!this->m_stor || !this->m_strides.empty()) {
        size_t total = this->total();
        this->m_stor = std::make_shared<TensorStorage>(total);
        this->m_offset = 0;
    }
}

void MindTensor::resize(const std::vector<int64_t> &new_shape) noexcept {
    this->m_shape = new_shape;
    this->m_strides = compute_strides(this->m_shape);

    size_t total = 1;
    for (const auto item : this->m_shape) total *= item;

    this->m_stor = std::make_shared<TensorStorage>(total);
    this->m_offset = 0;
}

void MindTensor::require_grad(const bool requires_grad) const noexcept {
    this->m_flow->requires_grad = requires_grad;
}

MindTensor MindTensor::flatten() const noexcept {
    const int64_t batch = this->m_shape[0];
    int64_t features = 1;

    for (int64_t i = 1; i < this->m_shape.size(); ++i) {
        features *= this->m_shape[i];
    }

    const std::vector _shape = {batch, features};

    MindTensor output(_shape, this->requires_grad());

    output.m_stor = this->m_stor;
    output.m_offset = this->m_offset;
    output.m_strides = compute_strides(_shape);

    return output;
}

MindTensor MindTensor::matmul(const MindTensor &other) const {
    err::IsAnError(this->m_shape.size() == 2, "cortex::_fw::MindTensor::matmul()", "Both tensors must be 2D for matrix multiplication");
    err::IsAnError(this->m_shape[1] == other.m_shape[0], "cortex::_fw::MindTensor::matmul()","Inner dimensions must match for matrix multiplication");

    const std::vector _shape = {this->m_shape[0], other.m_shape[1]};
    MindTensor output(_shape, this->requires_grad());

    avx2::matrix_t::matmul(this->data(), other.data(), output.data(), static_cast<size_t>(this->m_shape[0]), static_cast<size_t>(this->m_shape[1]), static_cast<size_t>(other.m_shape[1]));

    return output;
}

MindTensor MindTensor::transpose() const {
    err::IsAnError(this->m_shape.size() == 2, "cortex::_fw::MindTensor::transpose()", "Tensor must be 2D to transpose.");
    std::vector _shape = { this->m_shape[1], this->m_shape[0] };
    MindTensor output({ this->m_shape[1], this->m_shape[0] }, this->requires_grad());
    std::swap(output.m_strides[0], output.m_strides[1]);

    output.m_offset = this->m_offset;
    output.m_stor = this->m_stor;

    return output;
}

MindTensor MindTensor::permute(const std::vector<int64_t> &axes) const {
    err::IsAnError(axes.size() == this->m_shape.size(), "cortex::_fw::MindTensor::permute()", "Number of axes must match tensor dimensions");

    std::vector<int64_t> _shape(axes.size());

    for (size_t i = 0; i < axes.size(); ++i) {
        _shape[i] = this->m_shape[axes[i]];
    }

    for (const auto item : axes) {
        err::IsAnError(item >= 0 && item < static_cast<int64_t>(m_shape.size()), "cortex::_fw::MindTensor::permute()", "Invalid axis index");
    }

    MindTensor output(_shape, this->requires_grad());
    output.m_stor = this->m_stor;
    output.m_offset = this->m_offset;

    return output;
}

MindTensor MindTensor::slice(const int64_t dim, const int64_t start, const int64_t end) const {
    err::IsAnError(
        dim >= 0 && dim < static_cast<int64_t>(this->m_shape.size()),
        "cortex::_fw::MindTensor::slice()",
        "Dimension must be >= 0 and < 0"
    );
    err::IsAnError(
        start >= 0 && end <= this->m_shape[dim] && start < end,
        "cortex::_fw::MindTensor::slice()",
        "Invalid slice indices"
    );

    MindTensor output(this->m_shape, this->requires_grad());
    output.m_shape[dim] = end - start;
    output.m_offset = this->m_offset;
    output.m_stor = this->m_stor;

    return output;
}

MindTensor MindTensor::copy() const {
    MindTensor output(this->m_shape, this->requires_grad());
    const size_t total = this->total();
    std::memcpy(output.m_stor->ptr(), this->m_stor->ptr(), total * sizeof(float));
    return output;
}

MindTensor MindTensor::sum() const {
    MindTensor output({}, this->requires_grad());
    output.m_stor = std::make_shared<TensorStorage>(1);

    const float* it = this->m_stor->ptr() + this->m_offset;
    const size_t total = this->total();

    avx2::vec8f vt = avx2::zero();

    size_t i = 0;
    for (; i + 8 < total; i += 8) {
        const avx2::vec8f vd = avx2::load_u(it + i);
        vt = avx2::add(vt, vd);
    }
    float sum = 0;
    for (; i < total; ++i) sum += it[i];

    const float hs = avx2::hadd(vt) + sum;

    output.m_stor->ptr()[0] = hs;

    return output;
}

MindTensor MindTensor::sum(const int64_t dim, const bool keep) const {
    err::IsAnError(dim >= 0 && dim < static_cast<int64_t>(this->m_shape.size()), "cortex::_fw::MindTensor::sum()", "Dimension must be >= 0 and < 0");

    std::vector<int64_t> _shape;
    for (size_t i = 0; i < this->m_shape.size(); ++i) {
        if (i == static_cast<size_t>(dim)) {
            if (keep) _shape.push_back(1);
        } else {
            _shape.push_back(this->m_shape[i]);
        }
    }

    MindTensor output(_shape, this->requires_grad());

    const float* srcIt = this->m_stor->ptr() + this->m_offset;
    float* dstIt = this->m_stor->ptr();

    const size_t _dim_stride = this->m_strides[dim];
    size_t out = 1;
    for (size_t i = 0; i < dim; ++i) out *= this->m_shape[i];

    size_t in = 1;
    for (const long long i : this->m_shape) in *= i;
    const auto _dim_size = static_cast<float>(this->m_shape[dim]);

    for (size_t i = 0; i < out; ++i) {
        for (size_t j = 0; j < in; ++j) {
            const size_t dstIdx = i * in * j;
            avx2::vec8f vt = avx2::zero();
            size_t k = 0;
            for (; static_cast<float>(k) + 8 < _dim_size; k += 8) {
                const avx2::vec8f vd = avx2::load_u(srcIt + i * _dim_stride + k * in + j);
                vt = avx2::add(vt, vd);
            }
            float sum = 0;
            for (; static_cast<float>(k) < _dim_size; ++k) sum += srcIt[i * _dim_stride + k * in + j];

            const float hs = avx2::hadd(vt) + sum;
            dstIt[dstIdx] = hs;
        }
    }
    return output;
}

MindTensor MindTensor::mean(const int64_t dim, const bool keep) const {
    MindTensor output = this->sum(dim, keep);

    const auto divisor = static_cast<float>(this->m_shape[dim]);
    const size_t total = this->total();
    float* it = output.m_stor->ptr() + output.m_offset;

    size_t i = 0;
    const avx2::vec8f vd = avx2::broadcast(divisor);

    for (; i + 8 <= total; i += 8) {
        const avx2::vec8f vt = avx2::load_u(it + i);
        avx2::vec8f vr = avx2::add(vt, vd);
        avx2::store_u(it + i, vr);
    }
    for (; i < total; ++i) it[i] /= divisor;

    return output;
}

MindTensor MindTensor::view(const std::vector<int64_t> &new_shape) const {
    size_t total = 1;
    for (const auto& item : new_shape) total *= item;

    err::IsAnError(total == this->total(), "cortex::_fw::MindTensor::view()", "Total number of elements must remain the same in view.");
    err::IsAnError(this->is_contiguous(), "cortex::_fw::MindTensor::view()", "Only contiguous tensors can be reshaped with view.");

    MindTensor output(new_shape, this->requires_grad());
    output.m_stor = this->m_stor;
    output.m_offset = this->m_offset;
    output.m_strides = compute_strides(new_shape);

    return output;
}

MindTensor MindTensor::sqrt() const {
    const size_t total = this->total();
    MindTensor output(this->m_shape, this->requires_grad());

    float* it = this->m_stor->ptr() + this->m_offset;
    for (size_t i = 0; i < total; ++i) {
        avx2::vec8f vt = avx2::load_u(it + i);
        vt = avx2::sqrt(vt);
        avx2::store_u(it + i, vt);
    }

    output.m_stor = this->m_stor;
    output.m_offset = this->m_offset;

    return output;
}

MindTensor MindTensor::unsqueeze(int64_t dim) const {
    std::vector<int64_t> _shape = this->m_shape;

    if (dim < 0) dim += static_cast<int64_t>(_shape.size()) + 1;

    _shape.insert(_shape.begin() + dim, 1);

    MindTensor output(this->m_shape, this->requires_grad());

    output.m_offset = this->m_offset;
    output.m_stor = this->m_stor;

    return output;
}

MindTensor MindTensor::squeeze(int64_t dim) const {
    std::vector<int64_t> _shape = this->m_shape;

    if (dim < 0) dim += static_cast<int64_t>(_shape.size());

    _shape.erase(_shape.begin() + dim);

    MindTensor output(_shape, this->requires_grad());

    output.m_offset = this->m_offset;
    output.m_stor = this->m_stor;

    return output;
}

MindTensor MindTensor::reshape(const std::vector<int64_t> &new_shape) const {
    size_t total = 1;
    for (const auto& item : new_shape) total *= item;

    err::IsAnError(total == this->total(), "cortex::_fw::MindTensor::reshape()", "Total number of elements must remain the same");
    err::IsAnError(this->is_contiguous(), "cortex::_fw::MindTensor::reshape_inplace()", "reshape_inplace only works on contiguous tensors");

    MindTensor output(new_shape, this->requires_grad());

    output.m_offset = this->m_offset;
    output.m_stor = this->m_stor;
    return output;
}

MindTensor MindTensor::repeat(const std::vector<int64_t> &repeats) const {
    err::IsAnError(repeats.size() == this->m_shape.size(), "cortex::_fw::MindTensor::repeat()", "Repeat dims must match tensor dims");
    err::IsAnError(this->m_shape.size() == 2, "cortex::_fw::MindTensor::repeat()", "Currently only supports 2D tensors");

    std::vector<int64_t> _shape;
    for (size_t i = 0; i < repeats.size(); ++i) {
        _shape.push_back(this->m_shape[i] * repeats[i]);
    }

    MindTensor output(_shape, this->requires_grad());

    for (size_t i = 0; i < this->m_shape[0]; ++i) {
        for (size_t j = 0; j < repeats[0]; ++j) {
            for (size_t k = 0; k < this->m_shape[1]; ++k) {
                for (size_t l = 0; l < repeats[1]; ++l) {
                    output.at(i * repeats[0] + j, k * repeats[1] + l) = this->at(i, k);
                }
            }
        }
    }

    return output;
}

MindTensor &MindTensor::grad() {
    return *this->m_grad;
}

const MindTensor &MindTensor::grad() const {
    return *this->m_grad;
}

MindTensor MindTensor::operator+(const MindTensor &other) const {
    err::IsAnError(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator+()", "Shapes must match for element-wise addition.");

    MindTensor output(this->m_shape, this->requires_grad() || other.requires_grad());
    const size_t total = this->total();
    const float* _x = this->m_stor->ptr() + this->m_offset;
    const float* _y = other.m_stor->ptr() + other.m_offset;
    float* it = output.m_stor->ptr() + output.m_offset;
    avx2::matrix_t::add(_x, _y, it, total);

    if (output.requires_grad()) {
        output.m_grad = std::make_shared<MindTensor>();

        if (this->requires_grad() && other.requires_grad()) output.m_flow->parents = {this->m_flow, other.m_flow};

        auto tx = this->copy();
        auto ty = other.copy();

        output.m_flow->backward = [tx, ty](MindTensor& self) {
            if (self.requires_grad()) return;

            for (const auto& item : self.m_flow->parents) {
                if (item == tx) {}
            }
        };
    }

    return output;
}

MindTensor MindTensor::operator-(const MindTensor &other) const {
    err::IsAnError(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator-()", "Shapes must match for element-wise subtraction.");

    MindTensor output(this->m_shape, this->requires_grad() || other.requires_grad());

    const size_t total = this->total();
    const float* _x = this->m_stor->ptr() + this->m_offset;
    const float* _y = other.m_stor->ptr() + other.m_offset;
    float* it = output.m_stor->ptr() + output.m_offset;
    avx2::matrix_t::sub(_x, _y, it, total);

    return output;
}

MindTensor MindTensor::operator*(const MindTensor &other) const {
    err::IsAnError(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator*()", "Shapes must match for element-wise multiplication.");

    MindTensor output(this->m_shape, this->requires_grad());

    const size_t total = this->total();
    const float* _x = this->m_stor->ptr() + this->m_offset;
    const float* _y = other.m_stor->ptr() + other.m_offset;
    float* it = output.m_stor->ptr() + output.m_offset;
    avx2::matrix_t::sub(_x, _y, it, total);

    return output;
}

MindTensor MindTensor::operator/(const MindTensor &other) const {
    err::IsAnError(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator/()", "Shapes must match for element-wise division.");

    MindTensor output(this->m_shape, this->requires_grad());

    const size_t total = this->total();
    const float* _x = this->m_stor->ptr() + this->m_offset;
    const float* _y = other.m_stor->ptr() + other.m_offset;
    float* it = output.m_stor->ptr() + output.m_offset;
    avx2::matrix_t::div(_x, _y, it, total);

    return output;
}

MindTensor& MindTensor::operator+=(const MindTensor& other) {
    err::IsAnError(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator+=()", "Shapes must match for element-wise addition.");
    const size_t total = this->total();

    float* _x = this->m_stor->ptr() + this->m_offset;
    const float* _y = other.m_stor->ptr() + other.m_offset;

    avx2::matrix_t::add(_x, _y, _x, total);
    return *this;
}

MindTensor& MindTensor::operator-=(const MindTensor& other) {
    err::IsAnError(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator-=()", "Shapes must match for element-wise subtraction.");
    const size_t total = this->total();

    float* _x = this->m_stor->ptr() + this->m_offset;
    const float* _y = other.m_stor->ptr() + other.m_offset;

    avx2::matrix_t::sub(_x, _y, _x, total);
    return *this;
}

MindTensor& MindTensor::operator*=(const MindTensor& other) {
    err::IsAnError(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator*=()", "Shapes must match for element-wise multiplication.");
    const size_t total = this->total();

    float* _x = this->m_stor->ptr() + this->m_offset;
    const float* _y = other.m_stor->ptr() + other.m_offset;

    avx2::matrix_t::mul(_x, _y, _x, total);
    return *this;
}

MindTensor& MindTensor::operator/=(const MindTensor& other) {
    err::IsAnError(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator/=()", "Shapes must match for element-wise division.");
    const size_t total = this->total();

    float* _x = this->m_stor->ptr() + this->m_offset;
    const float* _y = other.m_stor->ptr() + other.m_offset;

    avx2::matrix_t::div(_x, _y, _x, total);
    return *this;
}

MindTensor MindTensor::operator+(const float scalar) const {
    MindTensor output(this->m_shape, this->requires_grad());
    const size_t total = this->total();
    const float* _x = this->m_stor->ptr() + this->m_offset;
    float* it = output.m_stor->ptr() + output.m_offset;
    const avx2::vec8f vs = avx2::broadcast(scalar);
    size_t i = 0;
    for (; i + 8 <= total; i += 8) {
        const avx2::vec8f vd = avx2::load_u(_x + i);
        avx2::vec8f vr = avx2::add(vd, vs);
        avx2::store_u(it + i, vr);
    }
    for (; i < total; ++i) it[i] = _x[i] + scalar;

    return output;
}

MindTensor MindTensor::operator-(const float scalar) const {
    MindTensor output(this->m_shape, this->requires_grad());
    const size_t total = this->total();
    const float* _x = this->m_stor->ptr() + this->m_offset;
    float* it = output.m_stor->ptr() + output.m_offset;
    const avx2::vec8f vs = avx2::broadcast(scalar);
    size_t i = 0;
    for (; i + 8 <= total; i += 8) {
        const avx2::vec8f vd = avx2::load_u(_x + i);
        avx2::vec8f vr = avx2::sub(vd, vs);
        avx2::store_u(it + i, vr);
    }
    for (; i < total; ++i) {
        it[i] = _x[i] - scalar;
    }

    return output;
}

MindTensor MindTensor::operator*(const float scalar) const {
    MindTensor output(this->m_shape, this->requires_grad());
    const size_t total = this->total();
    const float* _x = this->m_stor->ptr() + this->m_offset;
    float* it = output.m_stor->ptr() + output.m_offset;
    const avx2::vec8f vs = avx2::broadcast(scalar);
    size_t i = 0;
    for (; i + 8 <= total; i += 8) {
        const avx2::vec8f vd = avx2::load_u(_x + i);
        avx2::vec8f vr = avx2::mul(vd, vs);
        avx2::store_u(it + i, vr);
    }
    for (; i < total; ++i) it[i] = _x[i] * scalar;

    return output;
}

MindTensor MindTensor::operator/(const float scalar) const {
    MindTensor output(this->m_shape, this->requires_grad());
    const size_t total = this->total();
    const float* _x = this->m_stor->ptr() + this->m_offset;
    float* it = output.m_stor->ptr() + output.m_offset;
    const avx2::vec8f vs = avx2::broadcast(scalar);
    size_t i = 0;
    for (; i + 8 <= total; i += 8) {
        const avx2::vec8f vd = avx2::load_u(_x + i);
        avx2::vec8f vr = avx2::div(vd, vs);
        avx2::store_u(it + i, vr);
    }
    for (; i < total; ++i) it[i] = _x[i] / scalar;

    return output;
}

MindTensor& MindTensor::operator+=(const float scalar) {
    const size_t total = this->total();
    float* it = this->m_stor->ptr() + this->m_offset;
    const avx2::vec8f vs = avx2::broadcast(scalar);
    size_t i = 0;
    for (; i + 8 <= total; i += 8) {
        const avx2::vec8f vd = avx2::load_u(it + i);
        avx2::vec8f vr = avx2::add(vd, vs);
        avx2::store_u(it + i, vr);
    }
    for (; i < total; ++i) it[i] += scalar;
    return *this;
}

MindTensor& MindTensor::operator-=(const float scalar) {
    const size_t total = this->total();
    float* it = this->m_stor->ptr() + this->m_offset;
    const avx2::vec8f vs = avx2::broadcast(scalar);
    size_t i = 0;
    for (; i + 8 <= total; i += 8) {
        const avx2::vec8f vd = avx2::load_u(it + i);
        avx2::vec8f vr = avx2::sub(vd, vs);
        avx2::store_u(it + i, vr);
    }
    for (; i < total; ++i) it[i] += scalar;
    return *this;
}

MindTensor& MindTensor::operator*=(const float scalar) {
    const size_t total = this->total();
    float* it = this->m_stor->ptr() + this->m_offset;
    const avx2::vec8f vs = avx2::broadcast(scalar);
    size_t i = 0;
    for (; i + 8 <= total; i += 8) {
        const avx2::vec8f vd = avx2::load_u(it + i);
        avx2::vec8f vr = avx2::mul(vd, vs);
        avx2::store_u(it + i, vr);
    }
    for (; i < total; ++i) it[i] += scalar;
    return *this;
}

MindTensor& MindTensor::operator/=(const float scalar) {
    const size_t total = this->total();
    float* it = this->m_stor->ptr() + this->m_offset;
    const avx2::vec8f vs = avx2::broadcast(scalar);
    size_t i = 0;
    for (; i + 8 <= total; i += 8) {
        const avx2::vec8f vd = avx2::load_u(it + i);
        avx2::vec8f vr = avx2::add(vd, vs);
        avx2::store_u(it + i, vr);
    }
    for (; i < total; ++i) it[i] += scalar;
    return *this;
}