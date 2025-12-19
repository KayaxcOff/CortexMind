//
// Created by muham on 11.12.2025.
//

#ifndef CORTEXMIND_DENSE_HPP
#define CORTEXMIND_DENSE_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    class Dense : public _fw::Layer {
    public:
        Dense(int in_size, int out_size);
        ~Dense() override;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
        std::string config() override;

        void update_weights(float lr);
    private:
        int INPUT_SIZE;
        int OUTPUT_SIZE;
    };
}

#endif //CORTEXMIND_DENSE_HPP