//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    std::cout << "=== Dense Layer Test ===" << std::endl;

    nn::Dense layer(3, 2, host);

    std::cout << "Layer: " << layer.name() << std::endl;

    tensor input({2, 3}, host, true);
    input.fill(1.0f);

    std::cout << "\nInput shape: (2, 3)" << std::endl;
    std::cout << "Input:\n" << input << std::endl;

    tensor output = layer.forward(input);

    std::cout << "\nWeight shape: (3, 2)" << std::endl;
    std::cout << "Weight:\n" << layer.getParameters()[0].get() << std::endl;

    std::cout << "\nBias shape: (1, 2)" << std::endl;
    std::cout << "Bias:\n" << layer.getParameters()[1].get() << std::endl;

    std::cout << "\nOutput shape: (2, 2)" << std::endl;
    std::cout << "Output:\n" << output << std::endl;

    tensor loss = output * output;
    loss.sum().backward();

    std::cout << "\nLoss (sum of output^2):\n" << loss << std::endl;

    auto params = layer.getParameters();
    auto grads = layer.getGradients();

    std::cout << "\nWeight gradient:\n" << grads[0].get() << std::endl;
    std::cout << "\nBias gradient:\n" << grads[1].get() << std::endl;

    std::cout << "\nInput gradient:\n" << input.grad() << std::endl;

    return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
=== Dense Layer Test ===
Layer: Dense (3, 2)

Input shape: (2, 3)
Input:
[[1, 1, 1],
 [1, 1, 1]]
AddBackward initialized

Weight shape: (3, 2)
Weight:
[[-0.466691, 0.31129],
 [0.521881, -0.74493],
 [0.559325, -0.578344]]

Bias shape: (1, 2)
Bias:
[[0, 0]]

Output shape: (2, 2)
Output:
[[0.614515, -1.01198],
 [0.614515, -1.01198]]
MulBackward initialized
SumBackward initialized
MulBackward initialized
MulBackward destroyed
MulBackward initialized
MulBackward destroyed
SumBackward destroyed

Loss (sum of output^2):
[[0.377628, 1.02411],
 [0.377628, 1.02411]]

Weight gradient:
[[4, 4],
 [4, 4],
 [4, 4]]

Bias gradient:
[[2, 2]]

Input gradient:
[[-2.42324, 1.74123, -0.112927],
 [-2.42324, 1.74123, -0.112927]]
MulBackward destroyed
AddBackward destroyed
[WARN]  [CortexMind\framework\Gradient\flow.cpp | 33] Gradient Flow is null so graph can't build
[WARN]  [CortexMind\framework\Gradient\flow.cpp | 33] Gradient Flow is null so graph can't build
[WARN]  [CortexMind\framework\Gradient\flow.cpp | 33] Gradient Flow is null so graph can't build
[WARN]  [CortexMind\framework\Gradient\flow.cpp | 33] Gradient Flow is null so graph can't build
[WARN]  [CortexMind\framework\Gradient\flow.cpp | 33] Gradient Flow is null so graph can't build
[WARN]  [CortexMind\framework\Gradient\flow.cpp | 33] Gradient Flow is null so graph can't build

Process finished with exit code 0

*/