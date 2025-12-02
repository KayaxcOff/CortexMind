//
// Created by muham on 2.12.2025.
//

#ifndef CORTEXMIND_DROPOUT_HPP
#define CORTEXMIND_DROPOUT_HPP

#include <CortexMind/framework/NetBase/layer.hpp>
#include <random>

namespace cortex::nn {
    class Dropout : public Layer {
    public:
        explicit Dropout(double _p);
        ~Dropout() override = default;

        tensor forward(tensor &input) override;
        tensor backward(tensor &grad_output) override;

        std::vector<tensor*> getGradients() override { return {}; }
        std::vector<tensor*> getParameters() override { return {}; }

        std::string config() override;

        void train();
        void eval();
    private:
        double p;
        double scale;
        tensor mask;
        bool is_training;
        std::default_random_engine generator;
        std::uniform_real_distribution<> distribution;
    };
}

#endif //CORTEXMIND_DROPOUT_HPP