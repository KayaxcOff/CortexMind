//
// Created by muham on 15.12.2025.
//

#include "CortexMind/net/ActivationFunc/LeakyReLU/leaky.hpp"
#include <CortexMind/framework/Tools/MathTools/math.hpp>

using namespace cortex::net;
using namespace cortex::_fw;
using namespace cortex;

tensor LeakyReLU::forward(const tensor &input) {
    this->output = input;
    TensorFn::leaky_relu(this->output);
    return this->output;
}

tensor LeakyReLU::backward(const tensor &grad_output) {
    tensor grad_input = grad_output;

    for (int i = 0; i < grad_input.batch(); ++i) {
        for (int j = 0; j < grad_input.channel(); ++j) {
            for (int k = 0; k < grad_input.width(); ++k) {
                for (int l = 0; l < grad_input.height(); ++l) {
                    grad_input.at(i,j,k,l) *= this->output.at(i,j,k,l) > 0 ? 1.0f : this->alpha;
                }
            }
        }
    }
    return grad_input;
}
