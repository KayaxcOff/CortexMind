//
// Created by muham on 19.03.2026.
//

#include "CortexMind/net/NeuralNetwork/leaky_relu.hpp"
#include <CortexMind/tools/device.hpp>
#include <CortexMind/core/Engine/AVX2/funcs.hpp>
#include <CortexMind/core/Engine/CUDA/activation.cuh>
#include <CortexMind/core/Graph/ops.hpp>
#include <CortexMind/core/Tools/err.hpp>
#include <type_traits>

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

LeakyReLU::LeakyReLU(const float32 alpha) : Layer(true, "LeakyReLU"), alpha(alpha) {}

LeakyReLU::~LeakyReLU() = default;

tensor LeakyReLU::forward(tensor &input) {
    if (input.device() == dev::host) {
        size_t i = 0;
        for (; i + 8 <= input.numel(); i += 8) {
            avx2::storeu(input.get() + i, avx2::leaky_relu(avx2::loadu(input.get() + i)));
        }
        for (; i < input.numel(); ++i) {
            input.get()[i] = input.get()[i] > 0 ? input.get()[i] : input.get()[i] * this->alpha;
        }
    } else if (input.device() == dev::cuda) {
        cuda::activation_t::leaky_relu(input.get(), this->alpha, input.numel());
    } else {
        CXM_ASSERT(false, "cortex::nn::LeakyReLU::forward()", "Invalid device");
    }

    if (input.grad_required()) {
        auto grad_fn = std::make_shared<meta::leaky_relu>(&input, this->alpha);
        input.set_flow(std::move(grad_fn));
    }

    return input;
}

std::vector<ref<tensor>> LeakyReLU::parameters() {
    return {};
}

std::vector<ref<tensor>> LeakyReLU::gradients() {
    return {};
}
