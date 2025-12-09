//
// Created by muham on 7.12.2025.
//

#ifndef CORTEXMIND_KERNEL_HPP
#define CORTEXMIND_KERNEL_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::_fw {
    class ConvKernel {
    public:
        ConvKernel(int out_c, int in_c, int k_h, int k_w, float initValue = 0.0f);
        ~ConvKernel();

        tensor apply(const tensor& input);
        tensor backward(const tensor& in, tensor& out);
    private:
        tensor weights;
    };
}

#endif //CORTEXMIND_KERNEL_HPP