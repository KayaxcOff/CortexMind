//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_DENSE_HPP
#define CORTEXMIND_DENSE_HPP

#include <CortexMind/framework/NetBase/layer.hpp>

namespace cortex::nn {
    class Dense : public Layer {
    public:
        Dense(size_t in, size_t out);
        ~Dense() override = default;

        tensor forward(tensor &input) override;
        tensor backward(tensor &grad_output) override;
        std::vector<tensor*> getGradients() override;
        std::vector<tensor*> getParameters() override;
        std::string config() override;
    private:
        tensor weights;
        tensor bias;

        tensor gradWeights;
        tensor gradBias;

        tensor cached_input;

        size_t in_size, out_size;
    };
}

#endif //CORTEXMIND_DENSE_HPP