//
// Created by muham on 30.11.2025.
//

#include "CortexMind/framework/Kernel/kernel.hpp"
#include <cmath>

using namespace cortex::tools;
using namespace cortex;

MindKernel::MindKernel(const size_t _size, const size_t _in, const size_t _out) : weights(this->out, this->size, this->size, true), biases(1, 1, this->out, true), gradWeights(this->out, this->size, this->size, false), gradBias(1, 1, this->out, false), size(_size), in(_in), out(_out){
    this->Init();
}

MindKernel::~MindKernel() = default;

void MindKernel::Init() {
    const double limit = std::sqrt(2.0 / (this->size * this->size * this->in));
    this->weights.uniform_rand(-limit, limit);
    this->biases.fill(0.0);
}

void MindKernel::zero_grad() {
    this->gradWeights.zero();
    this->gradBias.zero();
}