//
// Created by muham on 8.11.2025.
//

#ifndef CORTEXMIND_DENSE_HPP
#define CORTEXMIND_DENSE_HPP

#include <CortexMind/Mind/NeuralNetwork/layer.hpp>
#include <vector>

namespace cortex::nn {
    class Dense : public Layer {
    public:
        Dense(size in, size out);
        ~Dense() override;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;

        [[nodiscard]] tensor getParams() const override;
        [[nodiscard]] tensor getGrads() const override;
    private:
        std::vector<tensor> weights;
        std::vector<tensor> biases;
        std::vector<tensor> gradWeights;
        tensor gradBiases;
        tensor outputGrad;
        tensor lastInput;
    };
}

#endif //CORTEXMIND_DENSE_HPP