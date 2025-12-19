//
// Created by muham on 10.12.2025.
//

#ifndef CORTEXMIND_LAYER_HPP
#define CORTEXMIND_LAYER_HPP

#include <CortexMind/framework/Params/params.hpp>
#include <CortexMind/framework/Net/optim.hpp>
#include <string>

namespace cortex::_fw {
    class Layer {
    public:
        Layer() = default;
        virtual ~Layer() = default;

        virtual tensor forward(const tensor &input) = 0;
        virtual tensor backward(const tensor &grad_output) = 0;
        [[nodiscard]] virtual std::string config() = 0;
        virtual void register_params(Optimizer& optim_fn) {
            if (!this->weights.empty()) {
                optim_fn.add_param(&weights, &grad_biases);
            }
            if (!this->biases.empty()) {
                optim_fn.add_param(&biases, &grad_biases);
            }
        }
    protected:
        tensor input_cache;
        tensor weights, biases;
        tensor grad_weights, grad_biases;
    };
}

#endif //CORTEXMIND_LAYER_HPP