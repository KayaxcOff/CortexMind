//
// Created by muham on 28.11.2025.
//

#ifndef CORTEXMIND_SGD_HPP
#define CORTEXMIND_SGD_HPP

#include <CortexMind/Mind/OptimizerFunc/optimizer.hpp>

namespace cortex::optim {
    class StochasticGradient : public Optimizer {
    public:
        explicit StochasticGradient(const float64 lr) : Optimizer(lr) {};
        ~StochasticGradient() override = default;

        void step(tensor &weights, const tensor &gradients) override;

        [[nodiscard]] float64 get_lr() const override {return this->learning_rate;}
        void set_lr(const float64 lr) override {this->learning_rate = lr;}
    };
}

#endif //CORTEXMIND_SGD_HPP