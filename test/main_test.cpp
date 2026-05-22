//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iomanip>
#include <ios>
#include <iostream>
#include <vector>

using namespace cortex;

int main() {
    const std::vector x_data = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    const std::vector y_data = {0.0f, 0.0f, 0.0f, 1.0f};

    tensor X({4, 2}, x_data.data(), host);
    tensor Y({4, 1}, y_data.data(), host);

    nn::Dense hidden(2, 4, host);
    nn::Tanh tanh;
    nn::Dense output_layer(4, 1, host);
    nn::Sigmoid sigmoid;

    loss::MeanSquared mse;
    opt::StochasticGradient sgd(0.1f);

    auto params = hidden.getParameters();
    auto out_params = output_layer.getParameters();
    params.insert(params.end(), out_params.begin(), out_params.end());
    sgd.SetParams(params);

    constexpr int epochs = 10000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        tensor h  = hidden.forward(X);
        tensor h2 = tanh.forward(h);
        tensor out = output_layer.forward(h2);
        tensor y_pred = sigmoid.forward(out);
        tensor l = mse.forward(y_pred, Y);

        sgd.zero_grad();
        l.backward();
        sgd.update();

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << std::setw(4) << epoch
                     << " | Loss: " << std::fixed << std::setprecision(6)
                     << l.get()[0] << std::endl;
        }
    }

    tensor h   = hidden.forward(X);
    tensor h2  = tanh.forward(h);
    tensor out = output_layer.forward(h2);
    tensor pred = sigmoid.forward(out);

    for (int i = 0; i < 4; ++i) {
        std::cout << "  [" << std::fixed << std::setprecision(1)
                  << x_data[i*2] << ", " << x_data[i*2+1] << "]"
                  << " -> " << std::fixed << std::setprecision(4) << pred.get()[i]
                  << " (expected: " << y_data[i] << ")" << std::endl;
    }

    return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
Epoch    0 | Loss: 0.277051
Epoch  200 | Loss: 0.070468
Epoch  400 | Loss: 0.027518
Epoch  600 | Loss: 0.013923
Epoch  800 | Loss: 0.008538
Epoch 1000 | Loss: 0.005902
Epoch 1200 | Loss: 0.004406
Epoch 1400 | Loss: 0.003466
Epoch 1600 | Loss: 0.002830
Epoch 1800 | Loss: 0.002376
Epoch 2000 | Loss: 0.002038
Epoch 2200 | Loss: 0.001777
Epoch 2400 | Loss: 0.001572
Epoch 2600 | Loss: 0.001405
Epoch 2800 | Loss: 0.001269
Epoch 3000 | Loss: 0.001155
Epoch 3200 | Loss: 0.001058
Epoch 3400 | Loss: 0.000976
Epoch 3600 | Loss: 0.000904
Epoch 3800 | Loss: 0.000842
Epoch 4000 | Loss: 0.000787
Epoch 4200 | Loss: 0.000738
Epoch 4400 | Loss: 0.000695
Epoch 4600 | Loss: 0.000656
Epoch 4800 | Loss: 0.000621
Epoch 5000 | Loss: 0.000590
Epoch 5200 | Loss: 0.000561
Epoch 5400 | Loss: 0.000535
Epoch 5600 | Loss: 0.000511
Epoch 5800 | Loss: 0.000489
Epoch 6000 | Loss: 0.000468
Epoch 6200 | Loss: 0.000449
Epoch 6400 | Loss: 0.000432
Epoch 6600 | Loss: 0.000416
Epoch 6800 | Loss: 0.000401
Epoch 7000 | Loss: 0.000386
Epoch 7200 | Loss: 0.000373
Epoch 7400 | Loss: 0.000361
Epoch 7600 | Loss: 0.000349
Epoch 7800 | Loss: 0.000338
Epoch 8000 | Loss: 0.000328
Epoch 8200 | Loss: 0.000318
Epoch 8400 | Loss: 0.000309
Epoch 8600 | Loss: 0.000300
Epoch 8800 | Loss: 0.000292
Epoch 9000 | Loss: 0.000284
Epoch 9200 | Loss: 0.000277
Epoch 9400 | Loss: 0.000270
Epoch 9600 | Loss: 0.000263
Epoch 9800 | Loss: 0.000256
  [0.0, 0.0] -> 0.0004 (expected: 0.0000)
  [0.0, 1.0] -> 0.0171 (expected: 0.0000)
  [1.0, 0.0] -> 0.0167 (expected: 0.0000)
  [1.0, 1.0] -> 0.9793 (expected: 1.0000)

Process finished with exit code 0
*/