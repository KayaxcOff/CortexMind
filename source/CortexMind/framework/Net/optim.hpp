//
// Created by muham on 5.12.2025.
//

#ifndef CORTEXMIND_OPTIM_HPP
#define CORTEXMIND_OPTIM_HPP

#include <CortexMind/framework/Params/params.hpp>
#include <CortexMind/framework/ErrorSystem/debug.hpp>

namespace cortex::_fw {

    struct TensorParams {
        tensor* _weights;
        tensor* _grads;
    };

    class Optimizer {
    public:
        explicit Optimizer(const double _lr = 0.001) {
            this->learning_rate = _lr;
        }
        virtual ~Optimizer() = default;

        virtual void step() = 0;

        virtual void zero_grad() {
            for (auto&[_weights, _grads] : this->params_list) {
                if (_grads) _grads->zero();
                else SynapticNode::captureFault(true, "cortex::_fw::net::OptimizerFunc::zero_grad()", "Gradient tensor is null.");
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

#endif //CORTEXMIND_OPTIM_HPP