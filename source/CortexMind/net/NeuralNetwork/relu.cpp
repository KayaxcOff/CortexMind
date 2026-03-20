//
// Created by muham on 18.03.2026.
//

#include "CortexMind/net/NeuralNetwork/relu.hpp"
#include <CortexMind/tools/device.hpp>
#include <CortexMind/core/Engine/AVX2/funcs.hpp>
#include <CortexMind/core/Engine/CUDA/activation.cuh>
#include <CortexMind/core/Graph/ops.hpp>
#include <memory>
#include <type_traits>

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

ReLU::ReLU() : Layer(true, "ReLU") {}

ReLU::~ReLU() = default;

tensor ReLU::forward(tensor &input) {

    if (input.device() == dev::host) {
        size_t i = 0;
        for (; i + 8 <= input.numel(); i += 8) {
            const avx2::vec8f vx = avx2::relu(avx2::loadu(input.get() + i));
            avx2::storeu(input.get() + i, vx);
        }
        for (; i < input.numel(); ++i) {
            input.get()[i] = input.get()[i] > 0 ? input.get()[i] : 0.0f;
        }
    } else if (input.device() == dev::cuda) {
        cuda::activation_t::relu(input.get(), input.numel());
    } else {
        CXM_ASSERT(false, "cortex::nn::ReLU::forward()", "Invalid device");
    }

    if (input.grad_required()) {
        auto grad_fn = std::make_shared<meta::relu>(&input);
        input.set_flow(std::move(grad_fn));
    }

    return input;
}

std::vector<ref<tensor>> ReLU::parameters() {
    return {};
}

std::vector<ref<tensor>> ReLU::gradients() {
    return {};
}
