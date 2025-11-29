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

            for (size epoch = 0; epoch < epochNum; ++epoch) {
                float64 epochLoss = 0.0;


                for (size start = 0; start < numSamples; start += batchSize) {
                    const size end = std::min(start + batchSize, numSamples);


                    std::vector<tensor> batch_outputs;
                    batch_outputs.reserve(end - start);

                    for (size j = start; j < end; ++j) {
                        tensor output = X[j];
                        for (const auto &layer : this->layers) {
                            output = layer->forward(output);
                        }
                        batch_outputs.push_back(output);
                    }


                    std::vector<tensor> batch_grads;
                    batch_grads.reserve(batch_outputs.size());

                    for (size idx = 0; idx < batch_outputs.size(); ++idx) {
                        const size data_idx = start + idx;

                        tensor loss_tensor = this->loss_fn_->forward(batch_outputs[idx], Y[data_idx]);

                        float64 loss_value = 0.0;
                        for (size r = 0; r < loss_tensor.get_rows(); ++r) {
                            for (size c = 0; c < loss_tensor.get_cols(); ++c) {
                                loss_value += loss_tensor(r, c);
                            }
                        }
                        epochLoss += loss_value;

                        batch_grads.push_back(this->loss_fn_->backward(batch_outputs[idx], Y[data_idx]));
                    }


                    for (auto grad : batch_grads) {
                        for (auto it = this->layers.rbegin(); it != this->layers.rend(); ++it) {
                            grad = (*it)->backward(grad);
                        }
                    }

                    for (const auto& layer : this->layers) {
                        tensor params = layer->getParams();
                        tensor grads = layer->getGrads();

                        this->optim_fn_->step(params, grads);
                        layer->setParams(params);
                    }
                }

                std::cout << "Epoch " << epoch +  1 << " / " << epochNum << " - Loss: " << epochLoss / static_cast<float64>(numSamples) << std::endl;
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