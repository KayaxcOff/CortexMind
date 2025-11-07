//
// Created by muham on 3.11.2025.
//

#ifndef CORTEXMIND_DENSE_HPP
#define CORTEXMIND_DENSE_HPP

#include <vector>
#include "../layer.hpp"

namespace cortex::layer {
    class Dense final : public Layer {
    public:
        Dense(size_t in, size_t out);
        ~Dense() override;

        math::MindVector forward(const math::MindVector &input) override;
        math::MindVector backward(const math::MindVector& output_gradients) override;
        void update(double lr) override;

        std::vector<math::MindVector*> get_parameters() override;
        std::vector<math::MindVector*> get_gradients() override;
    private:
        size_t inputSize, outputSize;

        std::vector<math::MindVector> weights;
        math::MindVector biases;
        math::MindVector last_input;
        std::vector<math::MindVector> grad_weights;
        math::MindVector grad_bias;
        math::MindVector output_grad;
        double learning_rate;
    };
}

#endif //CORTEXMIND_DENSE_HPP