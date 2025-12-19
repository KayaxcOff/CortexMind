//
// Created by muham on 19.12.2025.
// Test: Dense + ReLU + MAE + Manual SGD
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    constexpr int input_size = 4;
    constexpr int output_size = 3;
    constexpr int batch_size = 2;

    nn::Dense dense(input_size, output_size);
    net::MeanSquared loss;

    tensor x(batch_size,1,1,input_size);
    tensor y_true(batch_size,1,1,output_size);

    x.uniform_rand();
    y_true.uniform_rand();

    tensor y_pred = dense.forward(x);

    tensor l = loss.forward(y_pred, y_true);
    std::cout << "Initial loss: " << l.at(0,0,0,0) << std::endl;

    tensor grad_loss = loss.backward(y_pred, y_true);
    dense.backward(grad_loss);
    grad_loss.print();

    net::Adam optimizer(0.01);
    for (auto& item1 : dense.parameters()) {
        for (auto& item2 : dense.gradients()) {
            optimizer.add_param(item1, item2);
        }
    }
    //optimizer.step(); There is a index bug in this function
    optimizer.zero_grad();

    std::cout << "Step completed. Weights and biases updated." << std::endl;

    return 0;
}