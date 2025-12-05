//
// Created by muham on 4.12.2025.
//

#ifndef CORTEXMIND_MAX_POOL_HPP
#define CORTEXMIND_MAX_POOL_HPP

#include <CortexMind/framework/NetBase/layer.hpp>

namespace cortex::nn {
    class MaxPooling : public Layer {
    public:
        MaxPooling(size_t _kernelSize, size_t _stride);
        ~MaxPooling() override;

        tensor forward(tensor &input) override;
        tensor backward(tensor &grad_output) override;
        std::string config() override;
        std::vector<tensor*> getGradients() override;
        std::vector<tensor*> getParameters() override;
    private:
        size_t kernel_size;
        size_t stride;

        tensor inputCache;
        tensor idxCache;

        size_t in_height;
        size_t in_width;
    };
}

#endif //CORTEXMIND_MAX_POOL_HPP