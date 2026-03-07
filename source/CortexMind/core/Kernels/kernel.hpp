//
// Created by muham on 26.02.2026.
//

#ifndef CORTEXMIND_CORE_KERNELS_KERNEL_HPP
#define CORTEXMIND_CORE_KERNELS_KERNEL_HPP

#include <CortexMind/tools/params.hpp>
#include <CortexMind/core/Tools/params.hpp>
#include <CortexMind/core/Engine/Memory/device.hpp>

namespace cortex::_fw {
    class Kernel {
    public:
        Kernel(i64 in_channel, i64 out_channel, i64 kernel_width, i64 kernel_height, i64 stride = 1, i64 padding = 0, bool _requires_grad = false, sys::device _dev = sys::device::host);

        [[nodiscard]]
        tensor forward(tensor& in);

        [[nodiscard]]
        tensor& getWeight();
        [[nodiscard]]
        tensor& getBias();

        [[nodiscard]]
        i64 getOutputHeight(i64 input_h) const noexcept;
        [[nodiscard]]
        i64 getOutputWidth(i64 input_w)  const noexcept;
    private:
        tensor weight;
        tensor bias;
        i64 INPUT_CHANNEL, OUTPUT_CHANNEL;
        i64 KERNEL_WIDTH, KERNEL_HEIGHT;
        i64 m_padding, m_stride;

        [[nodiscard]]
        tensor im2col(const tensor& input, i64 H_out, i64 W_out) const;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_KERNELS_KERNEL_HPP