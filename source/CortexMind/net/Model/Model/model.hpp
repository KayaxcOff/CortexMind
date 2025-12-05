//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_MODEL_HPP
#define CORTEXMIND_MODEL_HPP

#include <CortexMind/framework/NetBase/layer.hpp>
#include <CortexMind/framework/NetBase/activation.hpp>
#include <CortexMind/framework/NetBase/loss.hpp>
#include <CortexMind/framework/NetBase/optimizer.hpp>
#include <vector>
#include <memory>
#include <stdexcept>

namespace cortex::model {
    class Model {
    public:
        template<class T, class... Args>
        void add(Args... args) {
            this->layers_.emplace_back(std::make_unique<T>(args...));
        }

        template<class LossT, class OptimT, class ActivT>
        void compile(double _lr=0.001) {
            this->loss_fn_ = std::make_unique<LossT>();
            this->activ_fn_ = std::make_unique<ActivT>(_lr);
            this->optim_fn_ = std::make_unique<OptimT>(_lr);
        }

        void summary() const;

        void train(std::vector<tensor>& _x, std::vector<tensor>& _y, int epochs=1, int batchSize=32);

        void forget() const;

        [[nodiscard]] tensor predict(const tensor& input) const;
    private:
        std::vector<std::unique_ptr<nn::Layer>> layers_;
        std::unique_ptr<act::Activation> activ_fn_;
        std::unique_ptr<loss::Loss> loss_fn_;
        std::unique_ptr<optim::Optimizer> optim_fn_;
    };
}

#endif //CORTEXMIND_MODEL_HPP