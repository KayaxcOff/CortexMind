//
// Created by muham on 30.11.2025.
//

#include "CortexMind/net/ActivationFunc/LeakyReLU/leaky.hpp"

using namespace cortex::act;
using namespace cortex;

tensor LeakyReLU::forward(const tensor &input) {
    tensor output = input;

    for (double & i : output.get_data()) {
        if (i < 0) {
            constexpr double alpha = 0.01;
            i *= alpha;
        }
    }

    return output;
}

tensor LeakyReLU::backward(const tensor &input) {
    tensor output = input;

    for (double & i : output.get_data()) {
        if (i < 0) {
            constexpr double alpha = 0.01;
            i = alpha;
        } else {
            i = 1.0;
        }
    }

    return output;
}