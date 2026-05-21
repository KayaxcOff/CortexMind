//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iomanip>
#include <iostream>

using namespace cortex;
/*
int main() {
    std::cout << "=== XOR Training ===" << std::endl;

    const std::vector x_data = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    const std::vector y_data = {0.0f, 1.0f, 1.0f, 0.0f};

    tensor X({4, 2}, x_data.data(), host);
    tensor Y({4, 1}, y_data.data(), host);

    nn::Dense hidden(2, 4, host);
    nn::ReLU relu;
    nn::Dense output_layer(4, 1, host);

    loss::MeanSquared mse;
    opt::StochasticGradient sgd(0.01f);

    auto params = hidden.getParameters();
    auto out_params = output_layer.getParameters();
    params.insert(params.end(), out_params.begin(), out_params.end());
    sgd.SetParams(params);

    constexpr int epochs = 1000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        tensor h  = hidden.forward(X);
        tensor h2 = relu.forward(h);
        tensor out = output_layer.forward(h2);
        tensor l = mse.forward(out, Y);

        sgd.zero_grad();
        l.backward();
        sgd.update();

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << std::setw(4) << epoch
                     << " | Loss: " << l.get()[0] << std::endl;
        }
    }

    std::cout << "\n=== Predictions ===" << std::endl;
    tensor h   = hidden.forward(X);
    tensor h2  = relu.forward(h);
    tensor pred = output_layer.forward(h2);

    const std::vector expected = {0.0f, 1.0f, 1.0f, 0.0f};
    for (int i = 0; i < 4; ++i) {
        std::cout << "Input: [" << x_data[i*2] << ", " << x_data[i*2+1] << "]"
                  << " | Pred: " << std::fixed << std::setprecision(4)
                  << pred.get()[i]
                  << " | Expected: " << expected[i] << std::endl;
    }

    return 0;
}
*/
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
=== XOR Training ===
Epoch    0 | Loss: 0.64706
Epoch  100 | Loss: 0.247152
Epoch  200 | Loss: 0.215043
Epoch  300 | Loss: 0.190462
Epoch  400 | Loss: 0.168852
Epoch  500 | Loss: 0.148541
Epoch  600 | Loss: 0.12994
Epoch  700 | Loss: 0.11239
Epoch  800 | Loss: 0.0961702
Epoch  900 | Loss: 0.0815512

=== Predictions ===
Input: [0, 0] | Pred: 0.3512 | Expected: 0.0000
Input: [0.0000, 1.0000] | Pred: 0.7237 | Expected: 1.0000
Input: [1.0000, 0.0000] | Pred: 0.7965 | Expected: 1.0000
Input: [1.0000, 1.0000] | Pred: 0.1728 | Expected: 0.0000

Process finished with exit code 0
*/
/*
int main() {
    const tensor x({3, 4, 5});
    x.uniform();

    nn::Flatten flatten;

    std::cout << "Before Flatten:\n" << x << std::endl;
    std::cout << "After Flatten:\n" << flatten.forward(x) << std::endl;

    return 0;
}
*/
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
Before Flatten:
[[[0.604562, 0.547813, 0.155267, 0.0253699, 0.158871],
  [0.718733, 0.0744213, 0.085619, 0.0460752, 0.00410658],
  [0.705178, 0.415138, 0.20905, 0.789734, 0.569061],
  [0.588765, 0.941404, 0.198698, 0.302112, 0.177009]],
 [[0.455091, 0.853672, 0.847177, 0.218747, 0.548446],
  [0.42135, 0.446353, 0.174903, 0.243396, 0.399728],
  [0.813903, 0.282096, 0.898836, 0.50906, 0.568226],
  [0.994414, 0.169631, 0.885777, 0.791445, 0.363771]],
 [[0.235086, 0.766914, 0.61629, 0.415676, 0.787867],
  [0.450171, 0.711964, 0.0906217, 0.938298, 0.18488],
  [0.837717, 0.607492, 0.583456, 0.0444533, 0.678444],
  [0.808512, 0.953992, 0.0267473, 0.368951, 0.949087]]]
After Flatten:
[[0.604562, 0.547813, 0.155267, 0.0253699, 0.158871, 0.718733, 0.0744213, 0.085619, 0.0460752, 0.00410658, 0.705178, 0.4
15138, 0.20905, 0.789734, 0.569061, 0.588765, 0.941404, 0.198698, 0.302112, 0.177009],
 [0.455091, 0.853672, 0.847177, 0.218747, 0.548446, 0.42135, 0.446353, 0.174903, 0.243396, 0.399728, 0.813903, 0.282096,
 0.898836, 0.50906, 0.568226, 0.994414, 0.169631, 0.885777, 0.791445, 0.363771],
 [0.235086, 0.766914, 0.61629, 0.415676, 0.787867, 0.450171, 0.711964, 0.0906217, 0.938298, 0.18488, 0.837717, 0.607492,
 0.583456, 0.0444533, 0.678444, 0.808512, 0.953992, 0.0267473, 0.368951, 0.949087]]

Process finished with exit code 0
*/