//
// Created by muham on 5.12.2025.
//

#ifndef CORTEXMIND_LAYER_HPP
#define CORTEXMIND_LAYER_HPP

#include <CortexMind/framework/Params/params.hpp>
#include <CortexMind/framework/Net/activ.hpp>
#include <vector>
#include <string>
#include <memory>

namespace cortex::_fw {
    class Layer {
    public:
        explicit Layer(std::unique_ptr<ActivationFunc> _activ_fn) : activ_fn_(std::move(_activ_fn)) {}
        virtual ~Layer() = default;

        virtual tensor forward(const tensor& input) = 0;
        virtual tensor backward(const tensor& grad_output) = 0;
        [[nodiscard]] virtual std::string config() const = 0;
        virtual std::vector<std::reference_wrapper<tensor>> gradients() = 0;
        virtual std::vector<std::reference_wrapper<tensor>> parameters() = 0;
    protected:
        std::unique_ptr<ActivationFunc> activ_fn_;
        tensor weights, biases;
        tensor grad_weights, grad_biases;
        tensor input_cache;
    };
}

#endif //CORTEXMIND_LAYER_HPP
