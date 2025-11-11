//
// Created by muham on 8.11.2025.
//

#ifndef CORTEXMIND_MODEL_HPP
#define CORTEXMIND_MODEL_HPP

#include <CortexMind/Mind/NeuralNetwork/layer.hpp>
#include <CortexMind/Mind/LossFunc/loss.hpp>
#include <CortexMind/Mind/ActivationFunc/activation.hpp>
#include <CortexMind/Mind/OptimizerFunc/optimizer.hpp>
#include <CortexMind/Utils/params.hpp>
#include <iostream>
#include <memory>
#include <vector>

namespace cortex::model {
    class Model {
    public:
        template<typename  T, typename... Args>
        void add(Args... args) {
            layers.emplace_back(std::make_unique<T>(args...));
        }

        template<typename L, typename O, typename A>
        void compile(float64 lr) {
            loss_fn_ = std::make_unique<L>();
            optim_fn_ = std::make_unique<O>(lr);
            activation_fn_ = std::make_unique<A>();
        }

        void fit();

        void summary() const {
            for (auto &layer : layers) {
                std::cout << layer->get_config() << std::endl;
            }
        }

        tensor predict(const tensor &input);
    private:
        std::vector<std::unique_ptr<nn::Layer>> layers;
        std::unique_ptr<loss::Loss> loss_fn_;
        std::unique_ptr<optim::Optimizer> optim_fn_;
        std::unique_ptr<act::Activation> activation_fn_;
    };
}

#endif //CORTEXMIND_MODEL_HPP