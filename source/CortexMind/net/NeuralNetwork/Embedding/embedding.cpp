//
// Created by muham on 14.12.2025.
//

#include "CortexMind/net/NeuralNetwork/Embedding/embedding.hpp"
#include <CortexMind/framework/Tools/Debug/catch.hpp>
#include <CortexMind/framework/Core/AVX/funcs.hpp>
#include <iostream>

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Embedding::Embedding(const int _input_dim, const int _vocab_size) : input_dim(_input_dim), vocab_size(_vocab_size) {
    this->weights.allocate(1, 1, this->vocab_size, this->input_dim);
    this->grad_weights.allocate(1, 1, this->vocab_size, this->input_dim);

    const float limit = std::sqrt(1.0f / static_cast<float>(this->input_dim));
    this->weights.uniform_rand(-limit, limit);
    this->grad_weights.zero();
}

Embedding::~Embedding() = default;

tensor Embedding::forward(const tensor &input) {
    this->input_cache = input;

    const int batch = input.batch();
    const int seq_len = input.height();

    tensor output(batch, 1, seq_len, this->input_dim);

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            const int idx = static_cast<int>(input.at(i, 0, j, 0));

            if (idx < 0 || idx >= this->vocab_size) CXM_ASSERT(true, "Index out of bounds");

            size_t offset = 0;
            while (offset + 8 <= static_cast<size_t>(this->input_dim)) {
                const float* src = this->weights.raw_ptr(idx * this->input_dim + offset);
                float* dst = output.raw_ptr((i * seq_len + j) * this->input_dim + offset);

                const avx2::reg vec = avx2::load(src);
                avx2::store(dst, vec);

                offset += 8;
            }
            while (offset < static_cast<size_t>(this->input_dim)) {
                output.at(i, 0, j, static_cast<int>(offset)) = this->weights.at(0, 0, idx, static_cast<int>(offset));
                offset++;
            }
        }
    }
    return output;
}

tensor Embedding::backward(const tensor &grad_output) {
    const int batch = this->input_cache.batch();
    const int seq_len = this->input_cache.height();

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            const int idx = static_cast<int>(this->input_cache.at(i, 0, j, 0));

            if (idx < 0 || idx >= this->vocab_size) CXM_ASSERT(true, "Index out of bounds");

            size_t offset = 0;
            while (offset + 8 <= static_cast<size_t>(this->input_dim)) {
                const size_t grad_idx = idx * this->input_dim + offset;
                const size_t out_idx = (i * seq_len + j) * this->input_dim + offset;

                float* grad_ptr = this->grad_weights.raw_ptr(grad_idx);
                const float* out_grad_ptr = grad_output.raw_ptr(out_idx);

                const avx2::reg grad_vec = avx2::load(grad_ptr);
                const avx2::reg out_grad_vec = avx2::load(out_grad_ptr);
                const avx2::reg result = avx2::add(grad_vec, out_grad_vec);

                avx2::store(grad_ptr, result);

                offset += 8;
            }

            while (offset < static_cast<size_t>(this->input_dim)) {
                this->grad_weights.at(0, 0, idx, static_cast<int>(offset)) += grad_output.at(i, 0, j, static_cast<int>(offset));
                offset++;
            }
        }
    }
    return tensor();
}
