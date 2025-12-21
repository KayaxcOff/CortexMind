//
// Created by muham on 11.12.2025.
//

#ifndef CORTEXMIND_MODEL_HPP
#define CORTEXMIND_MODEL_HPP

#include <CortexMind/framework/Tools/MathTools/math.hpp>
#include <CortexMind/framework/Tools/Debug/catch.hpp>
#include <CortexMind/framework/Net/layer.hpp>
#include <CortexMind/framework/Net/activ.hpp>
#include <CortexMind/framework/Net/loss.hpp>
#include <CortexMind/framework/Net/optim.hpp>
#include <iostream>
#include <vector>
#include <memory>

namespace cortex::net {
    class Model {
    public:
        Model() : isValid(false) {}
        ~Model() = default;

        template<typename  T, typename... Args>
        void add(Args... args) {
            this->layers_.emplace_back(std::make_unique<T>(args...));
        }

        template<typename LossT, typename OptimT, typename ActivT>
        void compile(float _lr = 0.001) {
            static_assert(std::is_base_of_v<_fw::Loss, LossT>, "LossT must derive from _fw::Loss");
            static_assert(std::is_base_of_v<_fw::Optimizer, OptimT>, "OptimT must derive from _fw::Optimizer");
            static_assert(std::is_base_of_v<_fw::Activation, ActivT>, "ActivT must derive from _fw::Activation");

            this->loss_fn_ = std::make_unique<LossT>();
            this->optim_fn_ = std::make_unique<OptimT>(_lr);
            this->activ_fn_ = std::make_unique<ActivT>();
            this->isValid = true;
        }

        void summary() const {
            if (this->isValid) {
                for (const auto& item : this->layers_) {
                    std::cout << item->config() << std::endl;
                }
            } else {
                std::cout << "Model not compiled" << std::endl;
            }
        }

        void train(const std::vector<tensor>& feats, const std::vector<tensor>& targets, const int epochs = 1){
            if (feats.empty() || targets.empty()) {
                CXM_ASSERT(true, "Feature or target data is empty");
            }
            if (feats.size() != targets.size()) {
                CXM_ASSERT(true, "Number of features and targets must match");
            }
            if (!this->isValid) {
                CXM_ASSERT(true, "Model not compiled");
            }

            for (const auto& item : layers_) {
                auto params = item->parameters();
                auto grads  = item->gradients();
                for (size_t k = 0; k < params.size(); ++k) {
                    this->optim_fn_->add_param(params[k], grads[k]);
                }
            }

            for (int i = 0; i < epochs; ++i) {
                float epoch_loss = 0.0f;
                float epoch_acc  = 0.0f;

                for (size_t j = 0; j < feats.size(); ++j) {
                    this->optim_fn_->zero_grad();

                    tensor x = feats[j];
                    tensor y = targets[j];

                    for (const auto& item : layers_) {
                        x = item->forward(x);
                    }
                    x = this->activ_fn_->forward(x);

                    tensor loss = this->loss_fn_->forward(x, y);
                    epoch_loss += loss.at(0, 0, 0, 0);
                    epoch_acc += _fw::TensorFn::accuracy(x, y);

                    tensor grad = this->loss_fn_->backward(x, y);

                    grad = this->activ_fn_->backward(grad);
                    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                        grad = (*it)->backward(grad);
                    }

                    this->optim_fn_->step();
                }


                std::cout
                    << "Epoch [" << (i + 1) << "/" << epochs << "] "
                    << "Loss: " << (epoch_loss / static_cast<float>(feats.size()))
                    << " | Accuracy: " << (epoch_acc / static_cast<float>(feats.size())) * 100.0f
                    << "%" << std::endl;
            }
        }

        [[nodiscard]] tensor predict(const tensor& input) const {
            tensor output = input;
            for (const auto& item : this->layers_) {
                output = item->forward(output);
            }
            return output;
        }

    private:
        std::vector<std::unique_ptr<_fw::Layer>> layers_;
        std::unique_ptr<_fw::Activation> activ_fn_;
        std::unique_ptr<_fw::Loss> loss_fn_;
        std::unique_ptr<_fw::Optimizer> optim_fn_;
        bool isValid;
    };
}

#endif //CORTEXMIND_MODEL_HPP