//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iomanip>
#include <iostream>

using namespace cortex;

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