//
// Created by muham on 3.11.2025.
//

#ifndef CORTEXMIND_LAYER_HPP
#define CORTEXMIND_LAYER_HPP

#include "../../Utils/MathTools/vector/vector.hpp"

namespace cortex::layer {
    class Layer {
    public:
        Layer();

        virtual ~Layer();

        virtual math::MindVector forward(const math::MindVector &input) = 0;
        virtual math::MindVector backward(const math::MindVector &grad_output) = 0;

        virtual void update(double lr);

        virtual std::vector<math::MindVector*> get_parameters() { return {}; }
        virtual std::vector<math::MindVector*> get_gradients() { return {}; }
    };
}

#endif //CORTEXMIND_LAYER_HPP