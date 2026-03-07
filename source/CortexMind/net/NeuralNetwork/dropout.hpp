//
// Created by muham on 5.03.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_DROPOUT_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_DROPOUT_HPP

#include <CortexMind/core/Net/layer.hpp>

namespace cortex::nn {
    class Dropout : public _fw::Layer {
    public:
        explicit Dropout(float32 p = 0.5f);
        ~Dropout() override;

        tensor forward(tensor& input) override;
        std::vector<_fw::ref<tensor>> parameters() override;
        std::vector<_fw::ref<tensor>> gradients() override;
    private:
        float32 p;
        tensor mask;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_DROPOUT_HPP