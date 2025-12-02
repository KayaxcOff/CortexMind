//
// Created by muham on 30.11.2025.
//

#include <iostream>
#include <CortexMind/net/NeuralNetwork/Flatten/flatten.hpp>
#include <CortexMind/net/NeuralNetwork/Dense/dense.hpp>

using namespace cortex::nn;
using namespace cortex;

int main() {
    constexpr size_t batch_size = 2;
    constexpr size_t H = 3;
    constexpr size_t W = 4;
    constexpr size_t dense_out = 5;

    Flatten flatten;
    Dense dense(H * W, dense_out);

    tensor input(batch_size, H, W);
    input.uniform_rand(0.1, 1.0);

    std::cout << "=== INPUT ===" << std::endl;
    input.print();

    tensor flat = flatten.forward(input);
    tensor output = dense.forward(flat);

    std::cout << "\n=== FORWARD ===" << std::endl;
    std::cout << "Output shape: ";
    for (auto s : output.get_shape()) std::cout << s << " ";
    std::cout << std::endl;
    output.print();

    tensor grad_output(batch_size, 1, dense_out);
    grad_output.fill(1.0);

    tensor grad_dense = dense.backward(grad_output);
    tensor grad_input = flatten.backward(grad_dense);

    std::cout << "\n=== BACKWARD ===" << std::endl;
    std::cout << "grad_input shape: ";
    for (auto s : grad_input.get_shape()) std::cout << s << " ";
    std::cout << std::endl;
    grad_input.print();

    auto grads = dense.getGradients();
    std::cout << "\nGradients sizes:" << std::endl;
    std::cout << "gradWeights shape: ";
    for (auto s : grads[0]->get_shape()) std::cout << s << " ";
    std::cout << std::endl;
    std::cout << "gradBias shape: ";
    for (auto s : grads[1]->get_shape()) std::cout << s << " ";
    std::cout << std::endl;

    return 0;
}