//
// Created by muham on 11.03.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_NORM_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_NORM_HPP

#include <CortexMind/core/Net/layer.hpp>

namespace cortex::nn {
    class LayerNorm : public _fw::Layer {
    public:
        explicit
        LayerNorm(int64 dim, float32 eps = 1e-5f);
        ~LayerNorm() override;

        tensor forward(tensor &input) override;
        std::vector<_fw::ref<tensor>> parameters() override;
        std::vector<_fw::ref<tensor>> gradients() override;
    private:
        tensor gamma, beta;
        float32 epsilon;
        int64 normalized_dim;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_NORM_HPP