//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_FLATTEN_HPP
#define CORTEXMIND_FLATTEN_HPP

#include <CortexMind/framework/NetBase/layer.hpp>

namespace cortex::nn {
    class Flatten : public Layer {
    public:
        Flatten();
        ~Flatten() override = default;

        tensor forward(tensor &input) override;
        tensor backward(tensor &grad_output) override;

        std::vector<tensor*> getGradients() override { return {}; }
        std::vector<tensor*> getParameters() override { return {}; }

        std::string config() override;
    private:
        std::vector<size_t> originalShape;
        tensor reshapedInput;
    };
}

#endif //CORTEXMIND_FLATTEN_HPP