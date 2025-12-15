//
// Created by muham on 15.12.2025.
//

#include "CortexMind/net/ActivationFunc/Sigmoid/sigmoid.hpp"
#include <CortexMind/framework/Tools/MathTools/math.hpp>
#include <CortexMind/framework/Core/AVX/funcs.hpp>

using namespace cortex::net;
using namespace cortex::_fw;
using namespace cortex::_fw::avx2;
using namespace cortex;

Sigmoid::Sigmoid() = default;

Sigmoid::~Sigmoid() = default;

tensor Sigmoid::forward(const tensor &input) {
    this->output = input;
    TensorFn::sigmoid(this->output);
    return this->output;
}

tensor Sigmoid::backward(const tensor &input) {
    tensor grad_input(this->output.batch(), this->output.channel(), this->output.height(), this->output.width());

    const reg one = broadcast(1.0f);

    for (size_t i = 0; i < this->output.data().size(); i++) {
        const reg y = load(&this->output.data()[i][0]);
        const reg dy = load(&input.data()[i][0]);

        const reg grad = mul(dy, mul(y, sub(one, y)));
        store(&grad_input.data()[i][0], grad);
    }
    return grad_input;
}
