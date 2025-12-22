//
// Created by muham on 14.12.2025.
//

#ifndef CORTEXMIND_DROPOUT_HPP
#define CORTEXMIND_DROPOUT_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    class Dropout : public _fw::Layer {
    public:
        explicit Dropout(float _dr = 0.05, bool _is_train = true);
        ~Dropout() override = default;

        void set_train(bool isTraining);
        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
        std::string config() override;
        std::array<tensor *, 2> parameters() override;
        std::array<tensor *, 2> gradients() override;
    private:
        float dropout_rate;
        tensor mask;
        bool isTrain;
    };
}

#endif //CORTEXMIND_DROPOUT_HPP