//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_CONVOLUTION_2D_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_CONVOLUTION_2D_HPP

#include <CortexMind/framework/Net/layer.hpp>
#include <utility>

namespace cortex::nn {
    /**
     * @brief 2D Convolutional Layer.
     *
     * Applies a 2D convolution over an input signal composed of several input planes.
     *
     * Mathematically:
     *
     *     output[b, c_out, h, w] = bias[c_out] + Σ (weight[c_out, c_in, kh, kw] * input[b, c_in, h+kh, w+kw])
     */
    class Conv2D : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a 2D Convolution layer.
         *
         * @param in_channels   Number of input channels
         * @param out_channels  Number of output channels (number of filters)
         * @param kH            Kernel height
         * @param kW            Kernel width
         * @param sH            Stride height (default: 1)
         * @param sW            Stride width (default: 1)
         * @param pH            Padding height (default: 0)
         * @param pW            Padding width (default: 0)
         * @param device        Target computation device
         */
        Conv2D(int64 in_channels, int64 out_channels, int64 kH, int64 kW, int64 sH = 1, int64 sW = 1, int64 pH = 0, int64 pW = 0, _fw::sys::DeviceType device = _fw::sys::DeviceType::kHOST);
        ~Conv2D() override;

        /**
         * @brief Performs the forward pass of the convolution.
         *
         * @param input Input tensor of shape `(batch, in_channels, height, width)`
         * @return Output tensor of shape `(batch, out_channels, out_height, out_width)`
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of the layer.
         *
         * @return Vector containing weight and bias tensors
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * @return Vector containing weight.grad() and bias.grad()
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    private:
        tensor weight;
        tensor bias;

        int64 KERNEL_WIDTH, KERNEL_HEIGHT;
        int64 STRIDE_WIDTH, STRIDE_HEIGHT;
        int64 PADDING_WIDTH, PADDING_HEIGHT;

        /**
         * @brief Performs im2col transformation (image to column).
         *
         * Converts the input image into a matrix suitable for matrix multiplication.
         * This is an internal helper used in the forward pass.
         *
         * @return Column matrix for GEMM operation
         */
        [[nodiscard]]
        tensor im2col(const tensor& input) const;
        /**
         * @brief Computes output height and width for convolution.
         *
         * Formula:
         *     out = (in + 2*padding - kernel) / stride + 1
         */
        [[nodiscard]]
        std::pair<int64, int64> compute_output_size(int64 input_height, int64 input_width) const;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_CONVOLUTION_2D_HPP