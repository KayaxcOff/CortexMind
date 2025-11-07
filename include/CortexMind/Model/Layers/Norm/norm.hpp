//
// Created by muham on 3.11.2025.
//

#ifndef CORTEXMIND_FLATTEN_HPP
#define CORTEXMIND_FLATTEN_HPP

#include <vector>
#include <CortexMind/Model/Layers/layer.hpp>
#include "CortexMind/Utils/MathTools/vector/vector.hpp"

namespace cortex::layer {
    class Normalize final : public Layer {
    public:
        explicit Normalize(const std::vector<int>& _input);
        ~Normalize() override;

        math::MindVector forward(const math::MindVector &input) override;
        math::MindVector backward(const math::MindVector &grad_output) override;
        void update(double lr) override;

        std::vector<math::MindVector*> get_parameters() override;
        std::vector<math::MindVector*> get_gradients() override;
    private:
        std::vector<int> input;
    };
}

#endif //CORTEXMIND_FLATTEN_HPP