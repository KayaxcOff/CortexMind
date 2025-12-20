//
// Created by muham on 20.12.2025.
//

#ifndef CORTEXMIND_LINEAR_H
#define CORTEXMIND_LINEAR_H

#include <CortexMind/models/model.hpp>

namespace cortex::tin {
    class LinearRegression : public Model {
    public:
        explicit LinearRegression(int input_dim, float lr = 0.1f);
        ~LinearRegression() override;

        void fit(std::vector<tensor> &X, std::vector<tensor> &Y) override;
        tensor predict(const tensor &pred) override;
    private:
        tensor weights;
        float bias;
        float lr;
    };
}

#endif //CORTEXMIND_LINEAR_H