//
// Created by muham on 5.12.2025.
//

#ifndef CORTEXMIND_SEQ_HPP
#define CORTEXMIND_SEQ_HPP

#include <CortexMind/framework/NetBase/activation.hpp>
#include <CortexMind/framework/NetBase/layer.hpp>
#include <CortexMind/framework/NetBase/loss.hpp>
#include <CortexMind/framework/NetBase/optimizer.hpp>
#include <memory>

namespace cortex::model {
    class Sequential {
    public:
        Sequential(std::initializer_list<std::unique_ptr<nn::Layer>> _layers);
        ~Sequential() = default;

        template<class LossT, class OptimT, class ActivT>
        void compile(double _lr=0.001) {
            this->loss_fn_ = std::make_unique<LossT>();
            this->activ_fn_ = std::make_unique<ActivT>(_lr);
            this->optim_fn_ = std::make_unique<OptimT>(_lr);
        }

        void train(std::vector<tensor>& _x, std::vector<tensor>& _y, int epochs=1, int batchSize=32);
        [[nodiscard]] tensor predict(const tensor& input) const;
    private:
        std::initializer_list<std::unique_ptr<nn::Layer>> layers_;
        std::unique_ptr<act::Activation> activ_fn_;
        std::unique_ptr<loss::Loss> loss_fn_;
        std::unique_ptr<optim::Optimizer> optim_fn_;
    };
}

#endif //CORTEXMIND_SEQ_HPP