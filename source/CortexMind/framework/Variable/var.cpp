//
// Created by muham on 30.11.2025.
//

#include "CortexMind/framework/Variable/var.hpp"

#include <random>
#include <CortexMind/framework/Log/log.hpp>
#include <stdexcept>

using namespace cortex::fw;

MindTensor::MindTensor(size_t _batch, size_t _row, size_t _col, const bool _grad) : required_grad(_grad) {
    this->shape = {_batch, _row, _col};
    this->strides.resize(3);

    if (_col == 0 || _row == 0 || _batch == 0) {
        this->strides = {0, 0, 0};
        this->data.clear();
        return;
    }

    this->strides[2] = 1;
    this->strides[1] = _col;
    this->strides[0] = _row * _col;

    const size_t totalSize = _batch * _row * _col;
    this->data.resize(totalSize, 0.0);
}

MindTensor::MindTensor(const MindTensor &other) = default;

MindTensor &MindTensor::operator=(const MindTensor &other) {
    if (this != &other) {
        this->shape = other.shape;
        this->strides = other.strides;
        this->data = other.data;
        this->required_grad = other.required_grad;
    }
    return *this;
}

MindTensor::~MindTensor() = default;

size_t MindTensor::getIdx(const size_t batch, const size_t row, const size_t col) const {
    if (batch >= shape[0] || row >= shape[1] || col >= shape[2]) {
        log("Index out of bounds.");
        throw std::out_of_range("Index out of bounds.");
    }
    return batch * strides[0] + row * strides[1] + col * strides[2];
}

double& MindTensor::operator()(const size_t b, const size_t r, const size_t c) {
    const size_t idx = this->getIdx(b, r, c);
    return this->data[idx];
}

double MindTensor::operator()(const size_t b, const size_t r, const size_t c) const {
    const size_t idx = this->getIdx(b, r, c);
    return this->data[idx];
}

void MindTensor::zero() {
    std::ranges::fill(this->data, 0.0);
}

void MindTensor::fill(const double _value) {
    std::ranges::fill(this->data, _value);
}

void MindTensor::uniform_rand(const double min, const double max) {
    std::uniform_real_distribution dist(min, max);
    std::random_device rd;
    std::mt19937 gen(rd());
    for (auto &val : this->data) {
        val = dist(gen);
    }
}

void MindTensor::print() const {
    for (size_t b = 0; b < shape[0]; ++b) {
        log("Batch " + std::to_string(b) + ":");
        for (size_t r = 0; r < shape[1]; ++r) {
            std::string rowStr;
            for (size_t c = 0; c < shape[2]; ++c) {
                rowStr += std::to_string((*this)(b, r, c)) + " ";
            }
            log(rowStr);
        }
    }
}

MindTensor MindTensor::operator+(const double scalar) const {
    MindTensor result(*this);
    for (auto &val : result.data) {
        val += scalar;
    }
    return result;
}

MindTensor MindTensor::operator+(const MindTensor &other) const {
    if (this->shape != other.shape) {
        log("Shapes do not match for addition.");
        throw std::invalid_argument("Shapes do not match for addition.");
    }
    MindTensor result(this->shape[0], this->shape[1], this->shape[2], this->required_grad || other.required_grad);
    for (size_t i = 0; i < this->data.size(); ++i) {
        result.data[i] = this->data[i] + other.data[i];
    }
    return result;
}

MindTensor MindTensor::operator-(const double scalar) const {
    MindTensor result(*this);
    for (auto &val : result.data) {
        val -= scalar;
    }
    return result;
}

MindTensor MindTensor::operator-(const MindTensor &other) const {
    if (this->shape != other.shape) {
        log("Shapes do not match for subtraction.");
        throw std::invalid_argument("Shapes do not match for subtraction.");
    }
    MindTensor result(this->shape[0], this->shape[1], this->shape[2], this->required_grad || other.required_grad);
    for (size_t i = 0; i < this->data.size(); ++i) {
        result.data[i] = this->data[i] - other.data[i];
    }
    return result;
}

MindTensor MindTensor::operator*(const double scalar) const {
    MindTensor result(*this);
    for (auto &val : result.data) {
        val *= scalar;
    }
    return result;
}

MindTensor MindTensor::operator*(const MindTensor &other) const {
    if (this->shape != other.shape) {
        log("Shapes do not match for multiplication.");
        throw std::invalid_argument("Shapes do not match for multiplication.");
    }
    MindTensor result(this->shape[0], this->shape[1], this->shape[2], this->required_grad || other.required_grad);
    for (size_t i = 0; i < this->data.size(); ++i) {
        result.data[i] = this->data[i] * other.data[i];
    }
    return result;
}

MindTensor MindTensor::operator/(const double scalar) const {
    MindTensor result(*this);
    for (auto &val : result.data) {
        val /= scalar;
    }
    return result;
}

MindTensor MindTensor::operator/(const MindTensor &other) const {
    if (this->shape != other.shape) {
        log("Shapes do not match for division.");
        throw std::invalid_argument("Shapes do not match for division.");
    }
    MindTensor result(this->shape[0], this->shape[1], this->shape[2], this->required_grad || other.required_grad);
    for (size_t i = 0; i < this->data.size(); ++i) {
        if (other.data[i] == 0) {
            log("Division by zero encountered.");
            throw std::invalid_argument("Division by zero encountered.");
        }
        result.data[i] = this->data[i] / other.data[i];
    }
    return result;
}

MindTensor &MindTensor::operator+=(const double scalar) {
    for (MindTensor result(*this); auto &val : result.data) {
        val += scalar;
    }
    return *this;
}

MindTensor &MindTensor::operator+=(const MindTensor &other) {
    if (this->shape != other.shape) {
        log("Shapes do not match for addition assignment.");
        throw std::invalid_argument("Shapes do not match for addition assignment.");
    }
    for (size_t i = 0; i < this->data.size(); ++i) {
        this->data[i] += other.data[i];
    }
    return *this;
}

MindTensor &MindTensor::operator-=(const double scalar) {
    for (auto &val : this->data) {
        val -= scalar;
    }
    return *this;
}

MindTensor &MindTensor::operator-=(const MindTensor &other) {
    if (this->shape != other.shape) {
        log("Shapes do not match for subtraction assignment.");
        throw std::invalid_argument("Shapes do not match for subtraction assignment.");
    }
    for (size_t i = 0; i < this->data.size(); ++i) {
        this->data[i] -= other.data[i];
    }
    return *this;
}

MindTensor &MindTensor::operator*=(const double scalar) {
    for (auto &val : this->data) {
        val *= scalar;
    }
    return *this;
}

MindTensor &MindTensor::operator*=(const MindTensor &other) {
    if (this->shape != other.shape) {
        log("Shapes do not match for multiplication assignment.");
        throw std::invalid_argument("Shapes do not match for multiplication assignment.");
    }
    for (size_t i = 0; i < this->data.size(); ++i) {
        this->data[i] *= other.data[i];
    }
    return *this;
}

MindTensor &MindTensor::operator/=(const double scalar) {
    for (auto &val : this->data) {
        val /= scalar;
    }
    return *this;
}

MindTensor &MindTensor::operator/=(const MindTensor &other) {
    if (this->shape != other.shape) {
        log("Shapes do not match for division assignment.");
        throw std::invalid_argument("Shapes do not match for division assignment.");
    }
    for (size_t i = 0; i < this->data.size(); ++i) {
        if (other.data[i] == 0) {
            log("Division by zero encountered.");
            throw std::invalid_argument("Division by zero encountered.");
        }
        this->data[i] /= other.data[i];
    }
    return *this;
}

MindTensor MindTensor::matmul(const MindTensor &other) const {
    const size_t R_A = this->shape[1];
    const size_t C_A = this->shape[2];
    const size_t R_B = other.shape[1];
    const size_t C_B = other.shape[2];
    const size_t B_A = this->shape[0];
    const size_t B_B = other.shape[0];

    if (B_A != B_B && B_B != 1) {
        log("Batch sizes must match or the second tensor's batch size must be 1 for matmul.");
        throw std::invalid_argument("Invalid batch size for matrix multiplication.");
    }

    if (C_A != R_B) {
        log("Inner dimensions must match for matrix multiplication (C_A != R_B).");
        throw std::invalid_argument("Inner dimensions mismatch for matmul.");
    }

    MindTensor result(B_A, R_A, C_B, this->required_grad || other.required_grad);

    for (size_t i = 0; i < B_A; ++i) {
        for (size_t j = 0; j < R_A; ++j) {
            for (size_t k = 0; k < C_B; ++k) {
                double sum = 0.0;
                for (size_t m = 0; m < C_A; ++m) {
                    const size_t batchB = (B_B == 1) ? 0 : i;
                    sum += (*this)(i, j, m) * other(batchB, m, k);
                }
                result(i, j, k) = sum;
            }
        }
    }
    return result;
}

MindTensor MindTensor::transpose() const {
    const size_t B = this->shape[0];
    const size_t R = this->shape[1];
    const size_t C = this->shape[2];

    MindTensor result(B, C, R, this->required_grad);

    for (size_t b = 0; b < B; ++b) {
        for (size_t r = 0; r < R; ++r) {
            for (size_t c = 0; c < C; ++c) {
                result(b, c, r) = (*this)(b, r, c);
            }
        }
    }
    return result;
}

MindTensor MindTensor::reshape(const std::vector<size_t> &new_shape) const {
    size_t newSize = 1;
    for (const auto &dim : new_shape) {
        newSize *= dim;
    }

    if (newSize != this->data.size()) {
        log("Reshape failed: Total number of elements must remain the same.");
        throw std::invalid_argument("Reshape failed: element count mismatch.");
    }

    if (new_shape.size() != 3) {
        log("Reshape failed: MindTensor currently supports only 3 dimensions.");
        throw std::invalid_argument("Reshape failed: dimension count mismatch.");
    }

    MindTensor result(new_shape[0], new_shape[1], new_shape[2], this->required_grad);
    result.data = this->data;

    return result;
}