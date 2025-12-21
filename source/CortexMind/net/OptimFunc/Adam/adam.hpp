//
// Created by muham on 16.12.2025.
//

#ifndef CORTEXMIND_ADAM_HPP
#define CORTEXMIND_ADAM_HPP

#include <CortexMind/framework/Net/optim.hpp>

namespace cortex::net {
    class Adam : public _fw::Optimizer {
    public:
        explicit Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);
        ~Adam() override = default;

        void zero_grad() override;
        void step() override;
        void add_param(tensor *weights, tensor *gradients) override;
    private:
        double b1, b2;
        double epsilon;
        int t{};

        std::vector<tensor> m_tensors;
        std::vector<tensor> v_tensors;
    };
}

#endif //CORTEXMIND_ADAM_HPP