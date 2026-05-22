//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_CONVOLUTION_2D_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_CONVOLUTION_2D_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    class Conv2D : public _fw::LayerBase {
    public:
        Conv2D(int64 in_channels, int64 out_channels, int64 kH, int64 kW, int64 sH = 1, int64 sW = 1, int64 pH = 0, int64 pW = 0, _fw::sys::DeviceType device = _fw::sys::DeviceType::kHOST);
        ~Conv2D() override;

        [[nodiscard]]
        tensor forward(const tensor &input) override;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    private:
        tensor kernel;
        tensor bias;

        int64 KERNEL_WIDTH, KERNEL_HEIGHT;
        int64 STRIDE_WIDTH, STRIDE_HEIGHT;
        int64 PADDING_WIDTH, PADDING_HEIGHT;

        [[nodiscard]]
        tensor im2col();
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_CONVOLUTION_2D_HPP