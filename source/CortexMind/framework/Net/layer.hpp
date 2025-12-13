//
// Created by muham on 10.12.2025.
//

#ifndef CORTEXMIND_LAYER_HPP
#define CORTEXMIND_LAYER_HPP

#include <CortexMind/framework/Params/params.hpp>
#include <string>

namespace cortex::_fw {
    class Layer {
    public:
        Layer() = default;
        virtual ~Layer() = default;

        virtual tensor forward(const tensor &input) = 0;
        virtual tensor backward(const tensor &grad_output) = 0;
        [[nodiscard]] virtual std::string config() = 0;
    protected:
        tensor input_cache;
        tensor weights, biases;
        tensor grad_weights, grad_biases;
    };
}

#endif //CORTEXMIND_LAYER_HPP