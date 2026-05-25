//
// Created by muham on 25.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_BATCH_NORM_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_BATCH_NORM_HPP

#include <CortexMind/framework/Net/layer.hpp>
#include <vector>

namespace cortex::nn {
    class BatchNormalization : public _fw::LayerBase {
    public:
        explicit BatchNormalization(const std::vector<int64>& axes, int64 num_features, float32 momentum = 0.1f, float32 epsilon = 1e-5f, _fw::sys::DeviceType device = _fw::sys::DeviceType::kHOST);
        ~BatchNormalization() override;

        [[nodiscard]]
        tensor forward(const tensor &input) override;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    private:
        int64 num_features;
        float32 momentum;
        float32 epsilon;

        tensor gamma;
        tensor beta;

        tensor running_mean;
        tensor running_var;

        std::vector<int64> axes;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_BATCH_NORM_HPP