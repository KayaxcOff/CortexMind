//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_OPTIMIZER_HPP
#define CORTEXMIND_OPTIMIZER_HPP

#include <CortexMind/framework/Params/params.hpp>
#include <CortexMind/framework/Log/log.hpp>
#include <vector>

namespace cortex::optim {

    struct TensorParams {
        tensor* _weights;
        tensor* _grads;
    };

    class Optimizer {
    public:
        explicit Optimizer(const double _lr) : learning_rate(_lr) {}
        virtual ~Optimizer() = default;

        virtual void step() = 0;
        virtual void zero_grad() {
            for (auto&[_weights, _grads] : params_list) {
                if (_grads) {
                    _grads->zero();
                } else {
                    log("Can't make grads zero");
                }
            }
        }
        virtual void register_parameters(std::vector<TensorParams>& params) {
            this->params_list = params;
        }
    protected:
        double learning_rate;
        std::vector<TensorParams> params_list;
    };
}

#endif //CORTEXMIND_OPTIMIZER_HPP