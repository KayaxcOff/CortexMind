//
// Created by muham on 8.11.2025.
//

#include <CortexMind/cortexmind.hpp>
#include <memory>
#include <vector>

#define EPOCH_NUM 10

using namespace cortex;

int main() {
    const auto neural_net_ = std::make_unique<model::Model>();

    const std::vector inputs {
        tensor(2, 2, 1.0),
        tensor(2, 2, 2.0),
        tensor(2, 2, 3.0)
    };
    const std::vector targets {
        tensor(1, 1, 0.0),
        tensor(1, 1, 1.0),
        tensor(1, 1, 0.0)
    };
    const auto test = tensor(2, 2, 1.0);

    neural_net_->add<nn::Dense>(2, 1);
    neural_net_->add<nn::Dense>(1, 2);
    neural_net_->compile<loss::MeanAbsolute, optim::StochasticGradient, act::Tanh>(0.001);

    neural_net_->fit(inputs, targets, EPOCH_NUM);

    const auto pred = neural_net_->predict(test);
    pred.print();

    return 0;
}