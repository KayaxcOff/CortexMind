//
// Created by muham on 15.12.2025.
//

#include "CortexMind/net/ActivationFunc/ReLU/relu.hpp"
#include <CortexMind/framework/Tools/MathTools/math.hpp>

using namespace cortex::net;
using namespace cortex::_fw;
using namespace cortex;

tensor ReLU::forward(const tensor &input) {
    this->output = input;
    TensorFn::relu(this->output);
    return this->output;
}

tensor ReLU::backward(const tensor &grad_output) {
    tensor grad_input = grad_output;

    for (int i = 0; i < grad_input.batch(); ++i) {
        for (int j = 0; j < grad_input.channel(); ++j) {
            for (int m = 0; m < grad_input.height(); ++m) {
                for (int n = 0; n < grad_input.width(); ++n) {
                    grad_input.at(i,j,m,n) = this->output.at(i,j,m,n) > 0 ? grad_output.at(i,j,m,n) : 0.0f;
                }
            }
        }
    }
    return grad_input;
}