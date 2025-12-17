//
// Created by muham on 16.12.2025.
//

#ifndef CORTEXMIND_BATCH_NORM_HPP
#define CORTEXMIND_BATCH_NORM_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    class BatchNorm : public _fw::Layer {
    public:
        explicit BatchNorm(float epsilon=1e-5, float momentum=0.1);
        ~BatchNorm() override;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
        std::string config() override;
    private:
        tensor gamma;
        tensor beta;
        tensor running_mean;
        tensor running_var;
        float momentum;
        float eps;
        tensor x_hat;
        tensor batch_mean;
        tensor batch_var;
    };
}

#endif //CORTEXMIND_BATCH_NORM_HPP