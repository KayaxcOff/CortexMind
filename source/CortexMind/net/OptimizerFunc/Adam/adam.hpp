//
// Created by muham on 3.12.2025.
//

#ifndef CORTEXMIND_ADAM_HPP
#define CORTEXMIND_ADAM_HPP

#include <CortexMind/framework/NetBase/optimizer.hpp>
#include <cmath>
#include <map>

namespace cortex::optim {
    class Adam : public Optimizer {
    public:
        explicit Adam(double learning_rate = 0.01f, double b1 = 0.9, double b2 = 0.999, double epsilon = 1e-8);
        ~Adam() override = default;

        void step() override;
    private:
        double beta1;
        double beta2;
        double epsilon;
        size_t t;

        std::vector<tensor> v_moments;
        std::vector<tensor> s_moments;

        void initialize_moments(const tensor& weights);
    };
}

#endif //CORTEXMIND_ADAM_HPP