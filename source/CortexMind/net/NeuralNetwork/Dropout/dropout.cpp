//
// Created by muham on 14.12.2025.
//

#include "CortexMind/net/NeuralNetwork/Dropout/dropout.hpp"
#include <CortexMind/framework/Core/AVX/funcs.hpp>
#include <CortexMind/framework/Tools/Debug/catch.hpp>
#include <iostream>

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Dropout::Dropout(const float _dr, const bool _is_train) : dropout_rate(_dr), isTrain(_is_train) {
    if (_dr < 0.0f || _dr >= 1.0f) {
        CXM_ASSERT(true, "Dropout parameter dr must be between 0.0 and 1.0");
    }
}

void Dropout::set_train(const bool isTraining) {
    this->isTrain = isTraining;
}

tensor Dropout::forward(const tensor &input) {
    if (!this->isTrain) {
        return input;
    }

    this->mask.allocate(input.batch(), input.channel(), input.height(), input.width());
    this->mask.uniform_rand();

    const float scale = 1.0f / (1.0f - this->dropout_rate);
    const size_t vec_size = this->mask.vec_size();

    const avx2::reg threshold = avx2::broadcast(this->dropout_rate);
    const avx2::reg scale_reg = avx2::broadcast(scale);
    const avx2::reg zero_reg = avx2::zero();

    for (size_t i = 0; i < vec_size; ++i) {
        avx2::reg mask_vec = avx2::load(&this->mask.dataIdx(i)[0]);

        const avx2::reg cmp = _mm256_cmp_ps(mask_vec, threshold, _CMP_GE_OQ);
        mask_vec = _mm256_blendv_ps(zero_reg, scale_reg, cmp);

        avx2::store(&this->mask.dataIdx(i)[0], mask_vec);
    }

    return input * this->mask;
}

tensor Dropout::backward(const tensor &grad_output) {
    if (!this->isTrain) {
        return grad_output;
    }

    return grad_output * this->mask;
}

std::string Dropout::config() {
    return "Dropout";
}

std::array<tensor *, 2> Dropout::parameters() {
    return {&this->weights, &this->biases};
}

std::array<tensor *, 2> Dropout::gradients() {
    return {&this->grad_weights, &this->grad_biases};
}