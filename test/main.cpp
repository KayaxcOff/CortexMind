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

    const std::vector<tensor> inputs;
    const std::vector<tensor> targets;
    const auto test = tensor({}, {});

    neural_net_->add<nn::Dense>(2, 1);
    neural_net_->add<nn::Dense>(1, 2);
    neural_net_->compile<loss::Loss, optim::Optimizer, act::Activation>(0.001);

    neural_net_->fit(inputs, targets, EPOCH_NUM);

    const auto pred = neural_net_->predict(test);
    pred.print();

    return 0;
}