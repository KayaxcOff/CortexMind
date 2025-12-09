//
// Created by muham on 5.12.2025.
//

#include "CortexMind/framework/Tensor/var.hpp"
#include <random>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace cortex::_fw;

MindTensor::MindTensor(const int batch, const int channels, const int height, const int width, const float initValue) : m_shape{0, 0, 0, 0}, m_size(0) {
    this->m_shape = {batch, channels, height, width};

    this->m_size = static_cast<size_t>(batch) * channels * height * width;
    const size_t blocks = (this->m_size + 7) / 8;

    this->m_data.resize(blocks);

    this->m_data.resize(blocks);
    for (auto &item : this->m_data) {
        item.resize(8, initValue);
    }
}

MindTensor::MindTensor(const MindTensor &other) : m_data(), m_shape(), m_size(0) {
    this->m_shape = other.m_shape;
    this->m_size = other.m_size;
    this->m_data.resize(other.m_data.size());

    for (size_t i = 0; i < this->m_data.size(); i++) {
        this->m_data[i] = other.m_data[i];
    }
}

float MindTensor::alignedIdx(const size_t idx) const {
    if (idx >= 8) SynapticNode::captureFault(true, "cortex::_fw::MindTensor::alignedIdx()", "Index out of bounds for alignedIdx().");
    float output = 0.0f;
    for (auto &item : this->m_data) {
        output += item[idx];
    }
    return output;
}

float &MindTensor::at(const int b, const int c, const int h, const int w) {
    const size_t idx = ((b * this->m_shape[1] + c) * this->m_shape[2] + h) * this->m_shape[3] + w;
    const size_t blk = idx / 8;
    const size_t offset = idx % 8;
    return this->m_data[blk][offset];
}

const float &MindTensor::at(const int b, const int c, const int h, const int w) const {
    const size_t idx = ((b * this->m_shape[1] + c) * this->m_shape[2] + h) * this->m_shape[3] + w;
    const size_t blk = idx / 8;
    const size_t offset = idx % 8;
    return this->m_data[blk][offset];
}

void MindTensor::uniform_rand(const float min, const float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(min,max);

    for(auto &item : this->m_data) {
        float tmp[8];
        for(float & i : tmp) i = dist(gen);
        avx::store(&item[0], avx::load(tmp));
    }
}

void MindTensor::zero() {
    const avx::reg z = avx::zero();
    for(auto &item : this->m_data) {
        avx::store(&item[0], z);
    }
}

void MindTensor::fill(const float value) {
    const avx::reg v = avx::broadcast(value);
    for(auto &item : this->m_data) {
        avx::store(&item[0], v);
    }
}

void MindTensor::print() {
    std::ostringstream oss;
    const std::vector shape(this->m_shape.begin(), this->m_shape.end());
    std::vector indices(4, 0);

    TensorRecursive(*this, shape, indices, 0, 0, oss);
    std::cout << oss.str() << std::endl;
}

MindTensor MindTensor::flatten() const {
    const int B = this->m_shape[0];
    const int C = this->m_shape[1];
    const int H = this->m_shape[2];
    const int W = this->m_shape[3];

    const int flat = C * H * W;
    //const size_t total = flat;

    MindTensor out(B, flat, 1, 1, 0.0f);

    size_t dst_idx = 0;

    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {

                int w = 0;
                while (w + 8 <= W) {
                    const float* src = &this->at(b, c, h, w);
                    float* dst = &out.at(0, static_cast<int>(dst_idx), 0, 0);
                    avx::store(dst, avx::load(src));

                    dst_idx += 8;
                    w += 8;
                }

                if (w < W) {
                    const int rem = W - w;
                    const float* src = &this->at(b, c, h, w);
                    float* dst = &out.at(0, static_cast<int>(dst_idx), 0, 0);

                    avx::store_partial(dst, avx::load_partial(src, rem), rem);

                    dst_idx += rem;
                }
            }
        }
    }

    return out;
}

/*
MindTensor MindTensor::flatten() const {
    const int batch = this->m_shape[0];
    const int channels = this->m_shape[1];
    const int height = this->m_shape[2];
    const int width = this->m_shape[3];

    const int flatSize = channels * height * width;
    const size_t total = static_cast<size_t>(batch) * flatSize;

    MindTensor out(batch, flatSize, 1, 1);

    float* dst = out.m_data[0].data();
    const float* src = this->m_data[0].data();

    size_t i = 0;

    while (i + 8 <= total) {
        avx::store(dst + i, avx::load(src + i));
        i += 8;
    }

    if (i < total) {
        const size_t rem = total - i;
        avx::store_partial(dst + i, avx::load_partial(src + i, rem), rem);
    }

    return out;
}
*/
MindTensor MindTensor::matmul(const MindTensor &other) const {
    MindTensor input = this->flatten();

    const int batch = input.shapeIdx(0);
    const int in_feats = input.shapeIdx(1);
    const int out_feats = other.shapeIdx(1);

    if (in_feats != other.shapeIdx(0)) SynapticNode::captureFault(true, "cortex::_fw::MindTensor::matmul()", "Incompatible shapes for matmul operation.");

    MindTensor output(batch, out_feats, 1, 1, 0.0f);

    for (int i = 0; i < batch; ++i) {
        const float* a_ptr = &input.at(i, 0, 0, 0);
        const float* b_ptr = &other.at(0, 0, 0, 0);
        float* out_ptr = &output.at(i, 0, 0, 0);

        avx::matmul_kernel(a_ptr, b_ptr, out_ptr, 1, out_feats, in_feats);
    }

    return output;
}

MindTensor MindTensor::transpose() const {
    return this->permute({0,2,3,1});
}

MindTensor MindTensor::permute(const std::array<int, 4> axes) const {
    std::array<bool, 4> seen{};
    for (int i = 0; i < 4; ++i) {
        if (axes[i] < 0 || axes[i] > 3)
            SynapticNode::captureFault(true, "MindTensor::permute()", "Axis out of bounds.");
        if (seen[axes[i]])
            SynapticNode::captureFault(true, "MindTensor::permute()", "Duplicate axis.");
        seen[axes[i]] = true;
    }

    std::array<int, 4> new_shape{};
    for (int i = 0; i < 4; ++i)
        new_shape[i] = this->m_shape[axes[i]];

    MindTensor output(new_shape[0], new_shape[1], new_shape[2], new_shape[3], 0.0f);

    for (int b2 = 0; b2 < new_shape[0]; ++b2) {
        for (int c2 = 0; c2 < new_shape[1]; ++c2) {
            for (int h2 = 0; h2 < new_shape[2]; ++h2) {

                int w2 = 0;
                while (w2 < new_shape[3]) {

                    const int remain = new_shape[3] - w2;
                    const int chunk = (remain >= 8 ? 8 : remain);

                    const int idx_out[4] = { b2, c2, h2, w2 };
                    const int idx_in[4]  = {
                        idx_out[axes[0]],
                        idx_out[axes[1]],
                        idx_out[axes[2]],
                        idx_out[axes[3]]
                    };

                    const float* src = &this->at(
                        idx_in[0],
                        idx_in[1],
                        idx_in[2],
                        idx_in[3]
                    );

                    float* dst = &output.at(b2, c2, h2, w2);

                    if (chunk == 8) {
                        avx::store(dst, avx::load(src));
                    } else {
                        avx::store_partial(dst, avx::load_partial(src, chunk), chunk);
                    }

                    w2 += chunk;
                }
            }
        }
    }

    return output;
}

/*
MindTensor MindTensor::permute(const std::array<int, 4> axes) const {
    std::array<bool,4> seen{};
    for(int i=0;i<4;++i){
        if(axes[i]<0 || axes[i]>3) SynapticNode::captureFault(true,"cortex::_fw::MindTensor::permute()","Axis out of bounds.");
        if(seen[axes[i]]) SynapticNode::captureFault(true,"cortex::_fw::MindTensor::permute()","Duplicate axis.");
        seen[axes[i]] = true;
    }

    std::array<int, 4> new_shape{};
    for(int i = 0; i < 4; ++i) new_shape[i] = this->m_shape[axes[i]];
    MindTensor output(new_shape[0], new_shape[1], new_shape[2], new_shape[3]);

    for(int b = 0; b < this->m_shape[0]; ++b){
        for(int c = 0; c < this->m_shape[1]; ++c){
            for(int h = 0; h < this->m_shape[2]; ++h){
                int w = 0;
                while(w + 8 <= this->m_shape[3]){
                    const auto v = avx::load(&this->at(b,c,h,w));
                    avx::store(&output.at(b,c,h,w), v);
                    w += 8;
                }
                if(w < this->m_shape[3]){
                    const size_t remaining = this->m_shape[3] - w;
                    const auto v = avx::load_partial(&this->at(b,c,h,w), remaining);
                    avx::store_partial(&output.at(b,c,h,w), v, remaining);
                }
            }
        }
    }

    return output;
}
*/

MindTensor &MindTensor::operator=(const MindTensor &other) {
    if (this == &other) return *this;
    this->m_shape = other.m_shape;
    this->m_data.resize(other.m_data.size());

    for(size_t i=0; i<other.m_data.size(); ++i) {
        avx::store(&this->m_data[i][0], avx::load(&other.m_data[i][0]));
    }
    return *this;
}

MindTensor MindTensor::operator+(const MindTensor &other) const {
    MindTensor output = *this;
    output += other;
    return output;
}

MindTensor MindTensor::operator-(const MindTensor &other) const {
    MindTensor output = *this;
    output -= other;
    return output;
}

MindTensor MindTensor::operator*(const MindTensor &other) const {
    MindTensor output = *this;
    output *= other;
    return output;
}

MindTensor MindTensor::operator/(const MindTensor &other) const {
    MindTensor output = *this;
    output /= other;
    return output;
}

MindTensor &MindTensor::operator+=(const MindTensor &other) {
    if(this->size() != other.size())
        SynapticNode::captureFault(true, "MindTensor::operator+=", "Size mismatch.");

    for(size_t i=0; i < this->m_data.size(); ++i)
        avx::add_kernel(&this->m_data[i][0], &other.m_data[i][0], &this->m_data[i][0], 8);

    return *this;
}

MindTensor &MindTensor::operator-=(const MindTensor &other) {
    if(this->size() != other.size())
        SynapticNode::captureFault(true, "MindTensor::operator-=", "Size mismatch.");

    for(size_t i=0; i < this->m_data.size(); ++i)
        avx::sub_kernel(&this->m_data[i][0], &other.m_data[i][0], &this->m_data[i][0], 8);

    return *this;
}

MindTensor &MindTensor::operator*=(const MindTensor &other) {
    if(this->size() != other.size())
        SynapticNode::captureFault(true, "MindTensor::operator*=", "Size mismatch.");

    for(size_t i=0; i < this->m_data.size(); ++i)
        avx::mul_kernel(&this->m_data[i][0], &other.m_data[i][0], &this->m_data[i][0], 8);

    return *this;
}

MindTensor &MindTensor::operator/=(const MindTensor &other) {
    if(this->size() != other.size())
        SynapticNode::captureFault(true, "MindTensor::operator/=", "Size mismatch.");

    for(size_t i=0; i < this->m_data.size(); ++i)
        avx::div_kernel(&this->m_data[i][0], &other.m_data[i][0], &this->m_data[i][0], 8);

    return *this;
}

MindTensor MindTensor::operator+(const float scalar) const {
    MindTensor output = *this;
    output += scalar;
    return output;
}

MindTensor MindTensor::operator-(const float scalar) const {
    MindTensor output = *this;
    output -= scalar;
    return output;
}

MindTensor MindTensor::operator*(const float scalar) const {
    MindTensor output = *this;
    output *= scalar;
    return output;
}

MindTensor MindTensor::operator/(const float scalar) const {
    MindTensor output = *this;
    output /= scalar;
    return output;
}

MindTensor &MindTensor::operator+=(const float scalar) {
    alignas(32) float tmp[8];
    for (float & i : tmp) i = scalar;

    for(auto& block : this->m_data)
        avx::add_kernel(block.data(), tmp, block.data(), 8);
    return *this;
}

MindTensor& MindTensor::operator-=(const float scalar) {
    alignas(32) float tmp[8];
    for (float & i : tmp) i = scalar;

    for(auto& block : this->m_data)
        avx::sub_kernel(block.data(), tmp, block.data(), 8);
    return *this;
}

MindTensor& MindTensor::operator*=(const float scalar) {
    alignas(32) float tmp[8];
    for (float & i : tmp) i = scalar;

    for(auto& block : this->m_data)
        avx::mul_kernel(block.data(), tmp, block.data(), 8);
    return *this;
}

MindTensor& MindTensor::operator/=(const float scalar) {
    alignas(32) float tmp[8];
    for (float & i : tmp) i = scalar;

    for(auto& block : this->m_data)
        avx::div_kernel(block.data(), tmp, block.data(), 8);
    return *this;
}

float *MindTensor::getIdx(const size_t flat_idx) {
    if (flat_idx >= this->m_size) {
        SynapticNode::captureFault(true, "cortex::_fw::MindTensor::getBlockPtr()", "Flat index out of bounds.");
        return nullptr;
    }
    const size_t blk = flat_idx / 8;
    const size_t offset = flat_idx % 8;
    return &m_data[blk][offset];
}

void MindTensor::TensorRecursive(const MindTensor &tensor, const std::vector<int> &shape, std::vector<int> &indices, const int dim, const int indent, std::ostream &os) {
    if (dim < 0 || dim >= static_cast<int>(shape.size())) return;

    const int cur_dim_size = shape[dim];
    const std::string spacing(indent, ' ');

    os << spacing << "[";

    if (dim == static_cast<int>(shape.size()) - 1) {
        for (int i = 0; i < cur_dim_size; ++i) {
            indices[dim] = i;
            os << std::fixed << std::setprecision(4)
               << tensor.at(indices[0], indices[1], indices[2], indices[3]);
            if (i != cur_dim_size - 1) os << ", ";
        }
    } else {
        os << "\n";
        for (int i = 0; i < cur_dim_size; ++i) {
            indices[dim] = i;
            TensorRecursive(tensor, shape, indices, dim + 1, indent + 2, os);
            if (i != cur_dim_size - 1) os << ",\n";
        }
        os << "\n" << spacing;
    }

    os << "]";
}
