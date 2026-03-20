//
// Created by muham on 19.03.2026.
//

#include "CortexMind/net/NeuralNetwork/input.hpp"
#include <CortexMind/core/Tools/err.hpp>

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Input::Input(const std::vector<int64> &shape) : Layer(true, "Input"), _shape(shape) {}

Input::Input(const std::initializer_list<int64> shape) : Layer(true, "Input"), _shape(shape) {}

tensor Input::forward(tensor &input) {
    CXM_ASSERT(this->_shape == input.shape(), "cortex::nn::Input::forward()", "Shapes mismatch");
    return input;
}

std::vector<ref<tensor>> Input::parameters() {
    return {};
}

std::vector<ref<tensor> > Input::gradients() {
    return {};
}
