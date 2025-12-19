//
// Created by muham on 19.12.2025.
// Test: Dense + ReLU + MAE + Manual SGD
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    _fw::MindKernel kernel(1, 1, 2, 2);

    tensor input(1, 1, 4, 4);
    input.fill(1.0);

    std::cout << "Input Tensor" << std::endl;
    input.print();

    const tensor output = kernel.apply(input);
    std::cout << "Output Tensor" << std::endl;
    output.print();

    const tensor grad_out(1, 1, output.height(), output.width(), 1.0f);

    const tensor grad_in = kernel.backward(input, grad_out);
    std::cout << "Grad Tensor" << std::endl;
    grad_in.print();

    return 0;
}