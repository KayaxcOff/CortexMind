//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>

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

Weight shape: (3, 2)
Weight:
[[-0.810244, -0.726305],
 [-0.912986, -0.466358],
 [-0.205836, 0.420748]]

Bias shape: (1, 2)
Bias:
[[0, 0]]

Output shape: (2, 2)
Output:
[[-1.92907, -0.771915],
 [-1.92907, -0.771915]]

Loss (sum of output^2):
[[3.72129, 0.595852],
 [3.72129, 0.595852]]

Weight gradient:
[[-7.71626, -3.08766],
 [-7.71626, -3.08766],
 [-7.71626, -3.08766]]

Bias gradient:
[[-7.71626, -3.08766]]

Input gradient:
[[4.24732, 4.2424, 0.144578],
 [4.24732, 4.2424, 0.144578]]

Process finished with exit code 0
*/