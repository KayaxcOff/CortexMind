//
// Created by muham on 10.12.2025.
//

#include "CortexMind/framework/Core/Tensor/tensor.hpp"
#include <CortexMind/framework/Core/AVX/funcs.hpp>
#include <CortexMind/framework/Core/AVX/matrix.hpp>
#include <CortexMind/framework/Tools/Debug/catch.hpp>
#include <iostream>
#include <random>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

MindTensor::MindTensor(const int batch, const int channel, const int height, const int width, const float value) : m_shape({batch, channel, height, width}) {
    this->m_size = static_cast<size_t>(batch) * channel * height * width;
    this->m_data.resize(this->m_size);
    for (auto& item : this->m_data) {
        item.fill(value);
    }
}

MindTensor::MindTensor(const MindTensor &other) = default;

MindTensor::~MindTensor() = default;

float &MindTensor::at(const int b, const int c, const int h, const int w) noexcept {
    const size_t idx = ((b * channel() + c) * height() + h) * width() + w;
    const size_t arrIdx = idx / 8;
    const size_t offset = idx % 8;
    return this->m_data[arrIdx][offset];
}

const float &MindTensor::at(const int b, const int c, const int h, const int w) const noexcept {
    const size_t idx = ((b * channel() + c) * height() + h) * width() + w;
    const size_t arrIdx = idx / 8;
    const size_t offset = idx % 8;
    return this->m_data[arrIdx][offset];
}

std::vector<AlignedArray<float, 8>> &MindTensor::data() noexcept {
    return this->m_data;
}

std::array<int, 4> MindTensor::shape() const noexcept {
    return this->m_shape;
}

size_t MindTensor::size() const noexcept {
    return this->m_size;
}

bool MindTensor::empty() const noexcept {
    return this->m_size == 0;
}

AlignedArray<float, 8> &MindTensor::dataIdx(const size_t idx) noexcept {
    return this->m_data[idx];
}

int MindTensor::batch() const noexcept {
    return this->m_shape[0];
}

int MindTensor::channel() const noexcept {
    return this->m_shape[1];
}

int MindTensor::height() const noexcept {
    return this->m_shape[2];
}

int MindTensor::width() const noexcept {
    return this->m_shape[3];
}

size_t MindTensor::vec_size() const {
    return (this->m_size + 7) / 8;
}

float *MindTensor::raw_ptr(const size_t idx) {
    const size_t arrIdx = idx / 8;
    const size_t offset = idx % 8;
    return &this->m_data[arrIdx][offset];
}

const float *MindTensor::raw_ptr(const size_t idx) const noexcept{
    const size_t arrIdx = idx / 8;
    const size_t offset = idx % 8;
    return &this->m_data[arrIdx][offset];
}

void MindTensor::print() const noexcept {
    if (this->empty()) {
        std::cout << "Empty Tensor" << std::endl;
    }

    std::cout << "Tensor shape: [" << this->batch() << ", " << this->channel() << ", " << this->height() << ", " << this->width() << "]" << std::endl;

    for (int i = 0; i < this->batch(); i++) {
        std::cout << "Batch " << i << ":" << std::endl;
        for (int j = 0; j < this->channel(); j++) {
            std::cout << " Channel " << j << ":" << std::endl;
            for (int k = 0; k < this->height(); k++) {
                std::cout << " [ ";
                for (int l = 0; l < this->width(); l++) {
                    std::cout << this->at(i, j, k, l) << " ";
                }
                std::cout << "]" << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

void MindTensor::uniform_rand(const float lower, const float upper) noexcept {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(lower, upper);

    for (auto& item : this->m_data) {
        float tmp[8];
        for(float & i : tmp) i = dist(gen);
        store(&item[0], load(tmp));
    }
}

void MindTensor::zero() noexcept {
    for (auto& item : this->m_data) {
        item.fill(0.0f);
    }
}

void MindTensor::fill(const float value) noexcept {
    for (auto& item : this->m_data) {
        item.fill(value);
    }
}

void MindTensor::allocate(const int batch, const int channel, const int height, const int width) noexcept {
    this->m_shape = {batch, channel, height, width};

    this->m_size = static_cast<size_t>(batch) * channel * height * width;
    const size_t num = (this->m_size + 7) / 8;
    this->m_data.resize(num);
}

MindTensor MindTensor::flatten() const noexcept {
    const size_t size = this->size();

    MindTensor result(1, 1, 1, static_cast<int>(size));

    for (size_t i = 0; i < this->m_data.size(); i++) {
        store(&result.dataIdx(i)[0], load(&this->m_data[i][0]));
    }
    return result;
}

MindTensor MindTensor::matmul(const MindTensor &other) const noexcept {
    if (this->width() != other.height())
        CXM_ASSERT(true, "Incompatible tensor shapes for matmul!");

    MindTensor result(this->batch(), this->channel(), this->height(), other.width());

    const size_t BC = this->batch() * this->channel();
    const size_t M = this->height();
    const size_t K = this->width();
    const size_t N = other.width();

    for (size_t bc = 0; bc < BC; ++bc) {
        matrix_t::matmul(&this->m_data[bc][0], &other.m_data[0][0], &result.m_data[bc][0], M, K, N);
    }

    return result;
}

MindTensor MindTensor::transpose() const noexcept {
    return this->permute({0, 1, 3, 2});
}

MindTensor MindTensor::permute(const std::array<int, 4> axes) const noexcept{
    std::array<bool, 4> used{};

    for (int i = 0; i < 4; i++) {
        if (axes[i] < 0 || axes[i] > 3) {
            CXM_ASSERT(true, "Invalid axis in permute!");
        }
        if (used[axes[i]]) {
            CXM_ASSERT(true, "Duplicate axis in permute!");
        }
        used[axes[i]] = true;
    }

    std::array<int, 4> newShape{};

    for (int i = 0; i < 4; i++) newShape[i] = this->m_shape[axes[i]];

    MindTensor output(newShape[0], newShape[1], newShape[2], newShape[3]);

    for (int i = 0; i < newShape[0]; ++i) {
        for (int j = 0; j < newShape[1]; ++j) {
            for (int k = 0; k < newShape[2]; ++k) {
                int sum = 0;

                while (sum < newShape[3]) {
                    const int remain = newShape[3] - sum;
                    const int step = remain >= 8 ? 8 : remain;

                    const int outIdx[4] = {i, j, k, sum};
                    const int inIdx[4] = {
                        outIdx[axes[0]],
                        outIdx[axes[1]],
                        outIdx[axes[2]],
                        outIdx[axes[3]]
                    };

                    const float* src = &this->at(inIdx[0], inIdx[1], inIdx[2], inIdx[3]);
                    float* dst = &output.at(i, j, k, sum);

                    if (step == 8) {
                        store(dst, load(src));
                    } else {
                        store_partial(dst, load(src), step);
                    }

                    sum += step;
                }
            }
        }
    }
    return output;
}

void MindTensor::operator()(const int b, const int c, const int h, const int w, const float value) noexcept {
    this->m_shape[0] =  b;
    this->m_shape[1] =  c;
    this->m_shape[2] =  h;
    this->m_shape[3] =  w;
    this->m_size = static_cast<size_t>(b) * c * h * w;
    const size_t num = (this->m_size + 7) / 8;
    this->m_data.resize(num);
    for (auto& item : this->m_data) {
        item.fill(value);
    }
}

MindTensor &MindTensor::operator=(const MindTensor &other) = default;

MindTensor MindTensor::operator+(const MindTensor &other) const noexcept {
    if (this->shape() != other.shape()) CXM_ASSERT(true, "Tensor shapes are not equal!");

    MindTensor result(this->batch(), this->channel(), this->height(), this->width());

    for (size_t i = 0; i < this->m_data.size(); i++) {
        store(&result.dataIdx(i)[0], add(load(&this->m_data[i][0]), load(&other.m_data[i][0])));
    }
    return result;
}

MindTensor MindTensor::operator-(const MindTensor &other) const noexcept {
    if (this->shape() != other.shape()) {
        return MindTensor();
    }

    MindTensor result(this->batch(), this->channel(), this->height(), this->width());

    for (size_t i = 0; i < this->m_data.size(); i++) {
        store(&result.dataIdx(i)[0], sub(load(&this->m_data[i][0]), load(&other.m_data[i][0])));
    }
    return result;
}

MindTensor MindTensor::operator*(const MindTensor &other) const noexcept {
    if (this->shape() != other.shape()) CXM_ASSERT(true, "Tensor shapes are not equal!");

    MindTensor result(this->batch(), this->channel(), this->height(), this->width());

    for (size_t i = 0; i < this->m_data.size(); i++) {
        store(&result.dataIdx(i)[0], mul(load(&this->m_data[i][0]), load(&other.m_data[i][0])));
    }
    return result;
}

MindTensor MindTensor::operator/(const MindTensor &other) const noexcept {
    if (this->shape() != other.shape()) CXM_ASSERT(true, "Tensor shapes are not equal!");

    MindTensor result(this->batch(), this->channel(), this->height(), this->width());

    for (size_t i = 0; i < this->m_data.size(); i++) {
        store(&result.dataIdx(i)[0], avx2::div(load(&this->m_data[i][0]), load(&other.m_data[i][0])));
    }
    return result;
}

MindTensor MindTensor::operator+(const float scalar) const noexcept {
    MindTensor result(this->batch(), this->channel(), this->height(), this->width());

    for (size_t idx = 0; idx < this->m_data.size(); idx++) {
        const reg v = load(&this->m_data[idx][0]);
        store(&result.dataIdx(idx)[0], add(v, broadcast(scalar)));
    }
    return result;
}

MindTensor MindTensor::operator-(const float scalar) const noexcept {
    MindTensor result(this->batch(), this->channel(), this->height(), this->width());

    for (size_t idx = 0; idx < this->m_data.size(); idx++) {
        const reg v = load(&this->m_data[idx][0]);
        store(&result.dataIdx(idx)[0], sub(v, broadcast(scalar)));
    }
    return result;
}

MindTensor MindTensor::operator*(const float scalar) const noexcept {
    MindTensor result(this->batch(), this->channel(), this->height(), this->width());

    for (size_t idx = 0; idx < this->m_data.size(); idx++) {
        const reg v = load(&this->m_data[idx][0]);
        store(&result.dataIdx(idx)[0], mul(v, broadcast(scalar)));
    }
    return result;
}

MindTensor MindTensor::operator/(const float scalar) const noexcept {
    MindTensor result(this->batch(), this->channel(), this->height(), this->width());

    for (size_t idx = 0; idx < this->m_data.size(); idx++) {
        const reg v = load(&this->m_data[idx][0]);
        store(&result.dataIdx(idx)[0], avx2::div(v, broadcast(scalar)));
    }
    return result;
}

MindTensor &MindTensor::operator+=(const MindTensor &other) noexcept {
    if (this->shape() != other.shape()) CXM_ASSERT(true, "Tensor shapes are not equal!");

    for (size_t i = 0; i < this->m_data.size(); i++) {
        matrix_t::add(&this->m_data[i][0], &other.m_data[i][0], &this->m_data[i][0], 8);
    }

    return *this;
}

MindTensor &MindTensor::operator-=(const MindTensor &other) noexcept {
    if (this->shape() != other.shape()) CXM_ASSERT(true, "Tensor shapes are not equal!");

    for (size_t i = 0; i < this->m_data.size(); i++) {
        matrix_t::sub(&this->m_data[i][0], &other.m_data[i][0], &this->m_data[i][0], 8);
    }

    return *this;
}

MindTensor &MindTensor::operator*=(const MindTensor &other) noexcept {
    if (this->shape() != other.shape()) CXM_ASSERT(true, "Tensor shapes are not equal!");

    for (size_t i = 0; i < this->m_data.size(); i++) {
        matrix_t::mul(&this->m_data[i][0], &other.m_data[i][0], &this->m_data[i][0], 8);
    }

    return *this;
}

MindTensor &MindTensor::operator/=(const MindTensor &other) noexcept {
    if (this->shape() != other.shape()) CXM_ASSERT(true, "Tensor shapes are not equal!");

    for (size_t i = 0; i < this->m_data.size(); i++) {
        matrix_t::div(&this->m_data[i][0], &other.m_data[i][0], &this->m_data[i][0], 8);
    }

    return *this;
}