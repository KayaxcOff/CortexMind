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

    constexpr int num_samples = 100;

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

    model.train(inputs, outputs, 10);

    return 0;
}

/*
----- output -----
Epoch [1/10] Loss: 0.449683 | Accuracy: 100%
Epoch [2/10] Loss: 0.434057 | Accuracy: 100%
Epoch [3/10] Loss: 0.418403 | Accuracy: 100%
Epoch [4/10] Loss: 0.404859 | Accuracy: 100%
Epoch [5/10] Loss: 0.394377 | Accuracy: 100%
Epoch [6/10] Loss: 0.385931 | Accuracy: 100%
Epoch [7/10] Loss: 0.378698 | Accuracy: 100%
Epoch [8/10] Loss: 0.373438 | Accuracy: 100%
Epoch [9/10] Loss: 0.369254 | Accuracy: 100%
Epoch [10/10] Loss: 0.366092 | Accuracy: 100%
*/