//
// Created by muham on 11.12.2025.
//

#ifndef CORTEXMIND_MODEL_HPP
#define CORTEXMIND_MODEL_HPP

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
        Model() = default;
        ~Model() = default;

        template<typename  T, typename... Args>
        void add(Args... args) {
            this->layers_.emplace_back(std::make_unique<T>(args...));
        }

        template<typename LoosT, typename OptimT, typename ActivT>
        void compile(float _lr) {
            this->loss_fn_ = std::make_unique<LoosT>();
            this->optim_fn_ = std::make_unique<OptimT>(_lr);
            this->activ_fn_ = std::make_unique<ActivT>();
        }

        void summary() const {
            for (const auto& item : this->layers_) {
                std::cout << item->config() << std::endl;
            }
        }

        void train(std::vector<tensor>& feats, std::vector<tensor>& targets, int epochs=1, int batch_size=1);

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
    };
}

#endif //CORTEXMIND_MODEL_HPP