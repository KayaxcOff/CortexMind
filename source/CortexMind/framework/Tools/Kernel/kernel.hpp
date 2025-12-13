//
// Created by muham on 10.12.2025.
//

#ifndef CORTEXMIND_KERNEL_HPP
#define CORTEXMIND_KERNEL_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::_fw {
    class MindKernel {
    public:
        MindKernel(int in_channel, int out_channel, int kernel_height, int kernel_width, float value);
        ~MindKernel();

        tensor apply(const tensor& input);
        tensor backward(const tensor& in, const tensor& grad_out);
    private:
        tensor weights;
        int IN_CHANNEL, OUT_CHANNEL;
        int KERNEL_HEIGHT, KERNEL_WIDTH;
    };
}

#endif //CORTEXMIND_KERNEL_HPP