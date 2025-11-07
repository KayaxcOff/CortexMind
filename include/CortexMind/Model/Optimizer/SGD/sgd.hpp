//
// Created by muham on 4.11.2025.
//

#ifndef CORTEXMIND_SGD_HPP
#define CORTEXMIND_SGD_HPP

#include <vector>
#include "CortexMind/Model/Optimizer/optimizer.hpp"
#include "../../../Utils/MathTools/vector/vector.hpp"

namespace cortex::optim {
    class StochasticGradient final : public Optimizer {
    public:
        StochasticGradient(std::vector<math::MindVector*> _params, std::vector<math::MindVector*> _grads, double _lr);

        void step() override;
    private:
        std::vector<math::MindVector*> params;
        std::vector<math::MindVector*> grads;
        double lr;
    };
}

#endif //CORTEXMIND_SGD_HPP