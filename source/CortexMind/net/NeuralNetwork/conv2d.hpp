//
// Created by muham on 26.02.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_CONV2D_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_CONV2D_HPP

#include <CortexMind/core/Net/layer.hpp>
#include <CortexMind/core/Kernels/kernel.hpp>
#include <memory>

namespace cortex::nn {
    /**
     * @brief 2D Convolutional neural network layer.
     *
     * Applies a 2D convolution operation over an input tensor using a set of
     * learnable kernels (filters).
     *
     * Given an input tensor of shape:
     *     (batch_size, in_channels, height, width)
     *
     * The layer produces an output tensor of shape:
     *     (batch_size, out_channels, out_height, out_width)
     *
     * where:
     *     out_height = floor((height + 2 * padding - kernel_height) / stride) + 1
     *     out_width  = floor((width  + 2 * padding - kernel_width ) / stride) + 1
     *
     * Each output channel is computed by convolving the corresponding kernel
     * across the spatial dimensions of the input.
     *
     * @note This layer performs a linear convolution operation.
     *       Non-linearity (e.g., ReLU) should be applied separately.
     */
    class Conv2D : public _fw::Layer {
    public:
        /**
         * @brief Constructs a 2D convolution layer.
         *
         * @param in_channel    Number of input channels.
         * @param out_channel   Number of convolution filters (output channels).
         * @param kernel_width  Width of the convolution kernel.
         * @param kernel_height Height of the convolution kernel.
         * @param stride        Stride of the convolution (default = 1).
         * @param padding       Zero-padding added to both spatial dimensions (default = 0).
         * @param _dev          Target device (e.g., host or cuda).
         */
        Conv2D(_fw::i64 in_channel, _fw::i64 out_channel, _fw::i64 kernel_width, _fw::i64 kernel_height, _fw::i64 stride = 1, _fw::i64 padding = 0, _fw::sys::device _dev = _fw::sys::device::host);
        /**
         * @brief Destructor.
         */
        ~Conv2D() override;

        /**
         * @brief Performs the forward pass of the convolution.
         *
         * Computes:
         *     output = Conv2D(input, kernel, stride, padding)
         *
         * @param input Input tensor of shape
         *              (batch_size, in_channels, height, width).
         *
         * @return Output tensor of shape
         *         (batch_size, out_channels, out_height, out_width).
         */
        tensor forward(tensor &input) override;
        /**
         * @brief Returns trainable parameters of the layer.
         *
         * Typically, includes:
         * - convolution kernels (filters)
         * - optional bias terms (if implemented inside kernel)
         *
         * @return Vector of references to parameter tensors.
         */
        std::vector<_fw::ref<_fw::MindTensor>> parameters() override;
        /**
         * @brief Returns gradients corresponding to trainable parameters.
         *
         * The order of gradients must match the order returned by parameters().
         *
         * @return Vector of references to gradient tensors.
         */
        std::vector<_fw::ref<_fw::MindTensor>> gradients() override;
    private:
        std::unique_ptr<_fw::Kernel> kernel_;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_CONV2D_HPP