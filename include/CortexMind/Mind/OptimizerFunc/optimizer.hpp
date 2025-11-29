//
// Created by muham on 8.11.2025.
//

#ifndef CORTEXMIND_OPTIMIZER_HPP
#define CORTEXMIND_OPTIMIZER_HPP

#include <CortexMind/Utils/params.hpp>

namespace cortex::optim {
    class Optimizer {
    public:
        explicit Optimizer(const float64 lr) : learning_rate(lr) {};
        virtual ~Optimizer() = default;

        virtual void step(tensor& weights, const tensor& gradients) = 0;

        [[nodiscard]] virtual float64 get_lr() const {return this->learning_rate;}
        virtual void set_lr(const float64 lr) {this->learning_rate = lr;}
    protected:
        float64 learning_rate;
    };
}

#endif //CORTEXMIND_OPTIMIZER_HPP