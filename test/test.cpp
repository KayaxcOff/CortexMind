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

    for(int i=0;i<batch_size;++i)
        for(int j=0;j<input_size;++j)
            x.at(i,0,0,j) = static_cast<float>(rand()) / RAND_MAX;

    for(int i=0;i<batch_size;++i)
        for(int j=0;j<output_size;++j)
            y_true.at(i,0,0,j) = static_cast<float>(rand()) / RAND_MAX;

    tensor y_pred = dense.forward(x);

    tensor l = loss.forward(y_pred, y_true);
    std::cout << "Initial loss: " << l.at(0,0,0,0) << std::endl;

    tensor grad_loss = loss.backward(y_pred, y_true);
    dense.backward(grad_loss);

    net::Adam optimizer(0.01);
    dense.register_params(optimizer);
    optimizer.step();
    optimizer.zero_grad();

    std::cout << "Step completed. Weights and biases updated." << std::endl;

    return 0;
}