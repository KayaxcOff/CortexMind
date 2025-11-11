//
// Created by muham on 8.11.2025.
//

#ifndef CORTEXMIND_LAYER_HPP
#define CORTEXMIND_LAYER_HPP

#include <CortexMind/Utils/params.hpp>
#include <string>

namespace cortex::nn {
    class Layer {
    public:
        Layer();
        virtual ~Layer();

        virtual tensor forward(const tensor& input) = 0;
        virtual tensor backward(const tensor& grad_output) = 0;

        [[nodiscard]] virtual tensor getParams() const = 0;
        [[nodiscard]] virtual tensor getGrads() const = 0;
        [[nodiscard]] virtual std::string get_config() const = 0;
    };
}

#endif //CORTEXMIND_LAYER_HPP