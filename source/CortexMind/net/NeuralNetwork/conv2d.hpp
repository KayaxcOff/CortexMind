//
// Created by muham on 26.02.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_CONV2D_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_CONV2D_HPP

#include <CortexMind/core/Net/layer.hpp>
#include <CortexMind/core/Kernels/kernel.hpp>
#include <memory>

namespace cortex::nn {
    class Conv2D : public _fw::Layer {
    public:

        Conv2D(int64 in_channel, int64 out_channel, int64 kernel_width, int64 kernel_height, int64 stride, int64 padding, _fw::sys::device _dev = _fw::sys::device::host);
        ~Conv2D() override;

        tensor forward(tensor &input) override;
        std::vector<_fw::ref<tensor>> parameters() override;
        std::vector<_fw::ref<tensor>> gradients() override;
    private:
        std::unique_ptr<_fw::Kernel> kernel_;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_CONV2D_HPP