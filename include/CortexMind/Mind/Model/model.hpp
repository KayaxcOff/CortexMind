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
            this->layers.emplace_back(std::make_unique<T>(args...));
        }

        template<typename L, typename O, typename A>
        void compile(float64 lr) {
            this->loss_fn_ = std::make_unique<L>();
            this->optim_fn_ = std::make_unique<O>(lr);
            this->activation_fn_ = std::make_unique<A>();
        }

        void fit(const std::vector<tensor>& X, const std::vector<tensor>& Y, const size epochNum = 1, const size batchSize = 32) {
            const size numSamples = X.size();

            if (X.size() != Y.size()) {
                throw std::invalid_argument("Input and output data size must be the same.");
            }

            for (size i = 0; i <epochNum; ++i) {
                float64 epochLoss = 0.0;

                for (size start=0; start < numSamples; ++start) {
                    const size end = std::min(start + batchSize, numSamples);

                    std::vector<tensor> batch_outputs;
                    for (size j = start; j < end; ++j) {
                        tensor output = X[j];
                        for (const auto &layer : this->layers) {
                            output = layer->forward(output);
                        }
                        batch_outputs.push_back(output);
                    }

                    std::vector<tensor> batch_grads;
                    for (size j = 0; j < batch_outputs.size(); ++j) {
                        epochLoss += this->loss_fn_->forward(batch_outputs[j], Y[start + j]);
                        batch_grads.push_back(this->loss_fn_->backward(batch_outputs[j], Y[start + j]));
                    }

                    for (auto grad : batch_grads) {
                        for (auto it = this->layers.rbegin(); it != this->layers.rend(); ++it) {
                            grad = (*it)->backward(grad);
                        }
                    }

                    for (const auto& it : this->layers) {
                        tensor params = it->getParams();
                        tensor grads = it->getGrads();

                        this->optim_fn_->step(params, grads);
                    }
                }
                std::cout << "Epoch " << i + 1
                  << " / " << epochNum
                  << " - Loss: " << epochLoss / static_cast<float64>(numSamples)
                  << std::endl;
            }
        }

        void summary() const {
            for (auto &layer : this->layers) {
                std::cout << layer->get_config() << std::endl;
            }
        }

        [[nodiscard]] tensor predict(const tensor &input) const {
            tensor output = input;
            for (const auto &layer : this->layers) {
                output = layer->forward(output);
            }
            return output;
        }
    private:
        std::vector<std::unique_ptr<nn::Layer>> layers;
        std::unique_ptr<loss::Loss> loss_fn_;
        std::unique_ptr<optim::Optimizer> optim_fn_;
        std::unique_ptr<act::Activation> activation_fn_;
    };
}

#endif //CORTEXMIND_MODEL_HPP