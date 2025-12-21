//
// Created by muham on 19.12.2025.
//

#include <CortexMind/cortexmind.hpp>
#include <vector>

using namespace cortex;

int main() {
    net::Model model;

    std::vector<tensor> inputs;
    std::vector<tensor> outputs;

    constexpr int num_samples = 10;

    for (int i = 0; i < num_samples; ++i) {
        tensor input;
        input.allocate(2, 1, 1, 4);
        input.uniform_rand();
        inputs.push_back(input);

        tensor output;
        output.allocate(2, 1, 1, 3);
        output.uniform_rand();
        outputs.push_back(output);
    }

    model.add<nn::Dense>(4, 3);

    model.compile<net::MeanAbsolute, net::Momentum, net::ReLU>();

    model.train(inputs, outputs, 5);

    return 0;
}