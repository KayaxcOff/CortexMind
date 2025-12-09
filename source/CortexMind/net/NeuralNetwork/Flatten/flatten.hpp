//
// Created by muham on 7.12.2025.
//

#ifndef CORTEXMIND_FLATTEN_HPP
#define CORTEXMIND_FLATTEN_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    class Flatten : public _fw::Layer {
    public:
        explicit Flatten(std::unique_ptr<_fw::ActivationFunc> activation_func);
        ~Flatten() override;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
        [[nodiscard]] std::string config() const override;
        std::vector<std::reference_wrapper<tensor>> gradients() override;
        std::vector<std::reference_wrapper<tensor>> parameters() override;
    private:
        std::array<int, 4> originalShape;
    };
}

#endif //CORTEXMIND_FLATTEN_HPP