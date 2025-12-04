//
// Created by muham on 4.12.2025.
//

#ifndef CORTEXMIND_MAX_POOL_HPP
#define CORTEXMIND_MAX_POOL_HPP

#include <CortexMind/framework/NetBase/layer.hpp>

namespace cortex::nn {
    class MaxPooling : public Layer {
    public:
        MaxPooling();
        ~MaxPooling() override;

        tensor forward(tensor &input) override;
        tensor backward(tensor &grad_output) override;
    };
}

#endif //CORTEXMIND_MAX_POOL_HPP