//
// Created by muham on 5.03.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_BATCH_NORM_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_BATCH_NORM_HPP

#include <CortexMind/core/Net/layer.hpp>

namespace cortex::nn {
    class BatchNorm1D : public _fw::Layer {
    public:
        explicit BatchNorm1D(int64 num_features, float32 eps = 1e-5f, float32 momentum = 0.1f);
        ~BatchNorm1D() override;

        tensor forward(tensor& input) override;
        std::vector<_fw::ref<tensor>> parameters() override;
        std::vector<_fw::ref<tensor>> gradients() override;
    private:
        int64 features;
        float32 momentum, eps;
        tensor gamma, beta;
        tensor running_mean, running_var;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_BATCH_NORM_HPP