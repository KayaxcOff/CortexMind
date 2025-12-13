//
// Created by muham on 10.12.2025.
//

#ifndef CORTEXMIND_OPTIM_HPP
#define CORTEXMIND_OPTIM_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::_fw {

    struct TensorParams {
        tensor* weights;
        tensor* gradients;
    };

    class Optimizer {
    public:
        explicit Optimizer(const double _lr = 0.001) {
            this->learning_rate = _lr;
        }
        virtual ~Optimizer() = default;

        virtual void step() = 0;
        virtual void zero_grad() {
            for (auto&[weights, gradients] : this->iters) {
                gradients->zero();
            }
        }
        virtual void add_param(tensor* weights, tensor* gradients) {
            this->iters.push_back({weights, gradients});
        }
    protected:
        double learning_rate;
        std::vector<TensorParams> iters;
    };
}

#endif //CORTEXMIND_OPTIM_HPP