//
// Created by muham on 9.03.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_MAX_POOL2D_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_MAX_POOL2D_HPP

#include <CortexMind/core/Net/layer.hpp>
#include <vector>

namespace cortex::nn {
    class MaxPooling2D : public _fw::Layer {
    public:
        MaxPooling2D(int64 kernel_size, int64 stride);
        ~MaxPooling2D() override;

        tensor forward(tensor &input) override;
        std::vector<_fw::ref<tensor>> parameters() override;
        std::vector<_fw::ref<tensor>> gradients() override;
    private:
        int64 KERNEL_SIZE;
        int64 STRIDE;
        std::vector<int64> maxIndices;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_MAX_POOL2D_HPP