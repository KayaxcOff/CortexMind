//
// Created by muham on 10.11.2025.
//

#ifndef CORTEXMIND_BATCH_NORM_HPP
#define CORTEXMIND_BATCH_NORM_HPP

#include <CortexMind/Mind/NeuralNetwork/layer.hpp>

namespace cortex::nn {
    class BatchNorm final : public Layer {
    public:
        BatchNorm(float64 _eps);
        ~BatchNorm() override;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;

        [[nodiscard]] tensor getParams() const override;
        [[nodiscard]] tensor getGrads() const override;
        [[nodiscard]] std::string get_config() const override;
    private:
        float64 eps;

        tensor gamma;
        tensor beta;

        tensor grad_gamma;
        tensor grad_beta;

        tensor x_norm;

        std::vector<float64> mean;
        std::vector<float64> var;
    };
}

#endif //CORTEXMIND_BATCH_NORM_HPP