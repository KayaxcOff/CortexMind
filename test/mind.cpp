//
// Created by muham on 30.11.2025.
//

#include <iostream>
#include <CortexMind/net/NeuralNetwork/Dense/dense.hpp>

using namespace cortex::nn;
using namespace cortex;

int main() {
    Dense layer(3, 2);


    tensor input(2, 1, 3);
    input.uniform_rand(0.1, 1.0);

    tensor output = layer.forward(input);

    std::cout << "Forward output:\n";
    for (size_t b = 0; b < 2; ++b) {
        for (size_t o = 0; o < 2; ++o) {
            std::cout << output(b, 0, o) << " ";
        }
        std::cout << std::endl;
    }

    tensor grad_output(2, 1, 2);
    grad_output.fill(1.0);

    tensor grad_input = layer.backward(grad_output);

    std::cout << "\nBackward grad_input:\n";
    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            std::cout << grad_input(b, 0, i) << " ";
        }
        std::cout << std::endl;
    }

    auto grads = layer.getGradients();
    std::cout << "\nGradients sizes:\n";
    std::cout << "gradWeights shape: " << grads[0]->get_shape()[0] << " "
              << grads[0]->get_shape()[1] << " " << grads[0]->get_shape()[2] << std::endl;
    std::cout << "gradBias shape: " << grads[1]->get_shape()[0] << " "
              << grads[1]->get_shape()[1] << " " << grads[1]->get_shape()[2] << std::endl;

    return 0;
}