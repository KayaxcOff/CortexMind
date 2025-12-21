//
// Created by muham on 10.12.2025.
//

#ifndef CORTEXMIND_KERNEL_HPP
#define CORTEXMIND_KERNEL_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::_fw {
    class MindKernel {
    public:
        MindKernel(int in_channel, int out_channel, int kernel_height, int kernel_width);
        ~MindKernel();

        tensor apply(const tensor& input);
        tensor backward(const tensor& input, const tensor& grad_output);
        void zero_grad() noexcept;
        std::array<tensor*, 1> parameters() noexcept;
        std::array<tensor*, 1> gradients() noexcept;
    private:
        tensor weights;
        tensor grad_weights;

        const int IN_CHANNEL;
        const int OUT_CHANNEL;
        const int KERNEL_HEIGHT;
        const int KERNEL_WIDTH;
    };
}

#endif //CORTEXMIND_KERNEL_HPP