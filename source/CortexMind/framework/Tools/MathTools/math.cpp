//
// Created by muham on 10.12.2025.
//

#include "CortexMind/framework/Tools/MathTools/math.hpp"
#include <CortexMind/framework/Core/AVX/funcs.hpp>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

void TensorFn::relu(tensor &input) {
    for (auto& item : input.data()) {
        const reg v = load(&item[0]);
        const reg zero = avx2::zero();
        const reg res = _mm256_max_ps(v, zero);
        store(&item[0], res);
    }
}

void TensorFn::sigmoid(tensor &input) {
    for (auto& item : input.data()) {
        const reg v = load(&item[0]);
        const reg neg_v = neg(v);
        const reg exp_val = exp_approx(neg_v);
        const reg one = broadcast(1.0f);
        const reg b = add(one, exp_val);
        const reg res = avx2::div(one, b);
        store(&item[0], res);
    }
}

void TensorFn::tanh(tensor &input) {
    for (auto& item : input.data()) {
        const reg v = load(&item[0]);
        const reg exp_pos = exp_approx(v);
        const reg exp_neg = exp_approx(neg(v));
        const reg numerator = sub(exp_pos, exp_neg);
        const reg denominator = add(exp_pos, exp_neg);
        const reg res = avx2::div(numerator, denominator);
        store(&item[0], res);
    }
}

void TensorFn::leaky_relu(tensor &input) {
    const reg alpha = broadcast(0.01f);
    for (auto& item : input.data()) {
        const reg v = load(&item[0]);
        const reg zero = avx2::zero();
        const reg mask = _mm256_cmp_ps(v, zero, _CMP_GT_OS);
        const reg scaled = mul(v, alpha);
        const reg res = _mm256_blendv_ps(scaled, v, mask);
        store(&item[0], res);
    }
}

void TensorFn::softmax(tensor &input) {
    for (auto& item : input.data()) {
        const reg max_val = load(&item[0]);
        const float max_f = h_sum(max_val);
        const reg max_bc = broadcast(max_f);

        reg sum = zero();
        reg tmp;
        for (int i = 0; i < 8; i+=8) {
            tmp = sub(load(&item[i]), max_bc);
            tmp = exp_approx(tmp);
            store(&item[i], tmp);
            sum = add(sum, tmp);
        }

        const float sum_f = h_sum(sum);
        const reg sum_bc = broadcast(sum_f);

        for (int i = 0; i < 8; i+=8) {
            tmp = load(&item[i]);
            store(&item[i], div(tmp, sum_bc));
        }
    }
}