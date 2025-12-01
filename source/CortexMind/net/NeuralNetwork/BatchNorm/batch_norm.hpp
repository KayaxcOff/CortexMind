//
// Created by muham on 1.12.2025.
//

#ifndef CORTEXMIND_BATCH_NORM_HPP
#define CORTEXMIND_BATCH_NORM_HPP

#include <CortexMind/framework/NetBase/layer.hpp>

namespace cortex::nn {
    class BatchNorm : public Layer {
    public:
        explicit BatchNorm(size_t num_feat, float eps = 1e-5f, float momentum = 0.1f);
        ~BatchNorm() override = default;

        tensor forward(tensor &input) override;
        tensor backward(tensor &grad_output) override;
        std::vector<tensor*> getGradients() override;
        std::vector<tensor*> getParameters() override;
        std::string config() override;

        void train();
        void eval();
    private:
        tensor gamma;
        tensor beta;

        tensor grad_gamma;
        tensor grad_beta;

        tensor running_mean;
        tensor running_var;

        double momentum;
        double eps;
        size_t num_feats;
        bool is_training;

        tensor cached_input;
        tensor cached_norm_input;
        tensor cached_variance;
        tensor cached_mean;
    };
}

#endif //CORTEXMIND_BATCH_NORM_HPP