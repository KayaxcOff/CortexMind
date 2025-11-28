//
// Created by muham on 28.11.2025.
//

#ifndef CORTEXMIND_SGD_HPP
#define CORTEXMIND_SGD_HPP

#include <CortexMind/Mind/OptimizerFunc/optimizer.hpp>

namespace cortex::optim {
    class StochasticGradient final : public Optimizer {
    public:
        explicit StochasticGradient(const float64 lr) : Optimizer(lr) {};
        ~StochasticGradient() override = default;

        void step(tensor &weights, const tensor &gradients) override;

        [[nodiscard]] float64 get_lr() const override;
        void set_lr(float64 lr) override;
    };
}

#endif //CORTEXMIND_SGD_HPP