//
// Created by muham on 28.11.2025.
//

#ifndef CORTEXMIND_ADAM_HPP
#define CORTEXMIND_ADAM_HPP

#include <CortexMind/Mind/OptimizerFunc/optimizer.hpp>

namespace cortex::optim {
    class Adam final : public Optimizer {
    public:
        Adam(float _lr, float64 _beta1, float64 _beta2, float64 _epsilon);

        void step(tensor &weights, const tensor &gradients) override;
    private:
        float64 beta1, beta2, epsilon;
        tensor m;
        tensor v;
        size t;
    };
}

#endif //CORTEXMIND_ADAM_HPP