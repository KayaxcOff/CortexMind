//
// Created by muham on 15.12.2025.
//

#include "CortexMind/net/ActivationFunc/ReLU/relu.hpp"
#include <CortexMind/framework/Tools/MathTools/math.hpp>
#include <CortexMind/framework/Tools/Debug/catch.hpp>
#include <iostream>

using namespace cortex::net;
using namespace cortex::_fw;
using namespace cortex;

tensor ReLU::forward(const tensor &input) {
    this->output = input;
    TensorFn::relu(this->output);
    return this->output;
}

tensor ReLU::backward(const tensor &grad_output) {
    if (grad_output.shape() != this->output.shape()) {
        CXM_ASSERT(true, "Backward grad output shape mismatch");
    }

    tensor grad_input = grad_output;

    for (int i = 0; i < grad_input.batch(); ++i) {
        for (int j = 0; j < grad_input.channel(); ++j) {
            for (int m = 0; m < grad_input.height(); ++m) {
                for (int n = 0; n < grad_input.width(); ++n) {
                    if (this->output.at(i,j,m,n) <= 0.0f) {
                        grad_input.at(i,j,m,n) = 0.0f;
                    }
                }
            }
        }
    }
    return grad_input;
}

/*
----- output -----
Epoch [1/5] Loss: 0.495405 | Accuracy: 100%
Epoch [2/5] Loss: 0.494967 | Accuracy: 100%
Epoch [3/5] Loss: 0.494237 | Accuracy: 100%
Epoch [4/5] Loss: 0.493365 | Accuracy: 100%
Epoch [5/5] Loss: 0.49236 | Accuracy: 100%
 */