//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_LAYER_HPP
#define CORTEXMIND_LAYER_HPP

#include <CortexMind/framework/Params/params.hpp>
#include <string>
#include <vector>

namespace cortex::nn {
    class Layer {
    public:
        Layer() = default;
        virtual ~Layer() = default;

        virtual tensor forward(tensor& input) = 0;
        virtual tensor backward(tensor& grad_output) = 0;
        [[nodiscard]] virtual std::vector<tensor*> getGradients() = 0;
        [[nodiscard]] virtual std::vector<tensor*> getParameters() = 0;
        [[nodiscard]] virtual std::string config() = 0;
    };
}

#endif //CORTEXMIND_LAYER_HPP