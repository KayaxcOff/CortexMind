//
// Created by muham on 29.05.2026.
//

#include "CortexMind/net/NeuralNetwork/softmax.hpp"
#include <CortexMind/framework/Engine/IX/activation.hpp>
#include <CortexMind/framework/Gradient/operations.hpp>
#include <memory>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Softmax::Softmax() : LayerBase("Softmax") {}

Softmax::~Softmax() = default;

tensor Softmax::forward(const tensor &input) {
    /*
    tensor output(input.shape(), input.device(), input.has_grad());

    ix::Activation::softmax(input.get(), output.get(), input.len(), input.device());

    if (output.has_grad()) [[likely]] {
        output.SetFlow(std::make_shared<meta::softmax>(input.pack(), output.pack()));
    }

    return output;
    */
    const i64 N = input.shape()[0];  // satır sayısı
    const i64 C = input.shape()[1];  // sınıf sayısı

    tensor output(input.shape(), input.device(), input.has_grad());

    const f32* in_ptr  = input.get();
    f32*       out_ptr = output.get();

    for (i64 n = 0; n < N; ++n) {
        ix::Activation::softmax(
            in_ptr  + n * C,
            out_ptr + n * C,
            static_cast<size_t>(C),
            input.device()
        );
    }

    if (output.has_grad()) [[likely]] {
        output.SetFlow(std::make_shared<meta::softmax>(input.pack(), output.pack()));
    }


    return output;
}

std::vector<ref<tensor>> Softmax::getParameters() {
    return {};
}

std::vector<ref<tensor>> Softmax::getGradients() {
    return {};
}