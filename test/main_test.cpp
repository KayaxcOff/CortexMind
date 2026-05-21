//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace cortex;

int main() {
    std::cout << "=== MAE vs MSE Comparison ===" << std::endl;

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
    nn::Tanh tanh1;
    nn::Dense output_layer(4, 1, host);
    nn::SigmoidFast sigmoid;

    loss::MeanAbsolute mae;
    loss::MeanSquared mse;
    opt::StochasticGradient sgd(0.01f);

    // ===== MAE Training =====
    std::cout << "\n--- MAE Training ---" << std::endl;

    auto params_mae = hidden.getParameters();
    auto out_params_mae = output_layer.getParameters();
    params_mae.insert(params_mae.end(), out_params_mae.begin(), out_params_mae.end());
    sgd.SetParams(params_mae);

    constexpr int epochs = 10000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        tensor h  = hidden.forward(X);
        tensor h2 = tanh1.forward(h);
        tensor out = output_layer.forward(h2);
        tensor y_pred = sigmoid.forward(out);
        tensor l = mae.forward(y_pred, Y);

        sgd.zero_grad();
        l.backward();
        sgd.update();

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << std::setw(4) << epoch
                     << " | Loss: " << std::fixed << std::setprecision(6)
                     << l.get()[0] << std::endl;
        }
    }

    std::cout << "\nMAE Predictions:" << std::endl;
    tensor h   = hidden.forward(X);
    tensor h2  = tanh1.forward(h);
    tensor out = output_layer.forward(h2);
    tensor pred_mae = sigmoid.forward(out);

    for (int i = 0; i < 4; ++i) {
        std::cout << "  [" << x_data[i*2] << ", " << x_data[i*2+1] << "]"
                  << " -> " << std::fixed << std::setprecision(4) << pred_mae.get()[i]
                  << " (expected: " << y_data[i] << ")" << std::endl;
    }

    return 0;
}
/*
with MSE:
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
=== MAE vs MSE Comparison ===

--- MAE Training ---
Epoch    0 | Loss: 0.277569
Epoch  200 | Loss: 0.266558
Epoch  400 | Loss: 0.258858
Epoch  600 | Loss: 0.253019
Epoch  800 | Loss: 0.248385
Epoch 1000 | Loss: 0.244557
Epoch 1200 | Loss: 0.241245
Epoch 1400 | Loss: 0.238222
Epoch 1600 | Loss: 0.235318
Epoch 1800 | Loss: 0.232408
Epoch 2000 | Loss: 0.229398
Epoch 2200 | Loss: 0.226218
Epoch 2400 | Loss: 0.222817
Epoch 2600 | Loss: 0.219155
Epoch 2800 | Loss: 0.215204
Epoch 3000 | Loss: 0.210941
Epoch 3200 | Loss: 0.206350
Epoch 3400 | Loss: 0.201420
Epoch 3600 | Loss: 0.196144
Epoch 3800 | Loss: 0.190521
Epoch 4000 | Loss: 0.184553
Epoch 4200 | Loss: 0.178252
Epoch 4400 | Loss: 0.171638
Epoch 4600 | Loss: 0.164740
Epoch 4800 | Loss: 0.157601
Epoch 5000 | Loss: 0.150274
Epoch 5200 | Loss: 0.142821
Epoch 5400 | Loss: 0.135313
Epoch 5600 | Loss: 0.127825
Epoch 5800 | Loss: 0.120430
Epoch 6000 | Loss: 0.113200
Epoch 6200 | Loss: 0.106197
Epoch 6400 | Loss: 0.099473
Epoch 6600 | Loss: 0.093070
Epoch 6800 | Loss: 0.087014
Epoch 7000 | Loss: 0.081325
Epoch 7200 | Loss: 0.076007
Epoch 7400 | Loss: 0.071059
Epoch 7600 | Loss: 0.066473
Epoch 7800 | Loss: 0.062233
Epoch 8000 | Loss: 0.058321
Epoch 8200 | Loss: 0.054719
Epoch 8400 | Loss: 0.051404
Epoch 8600 | Loss: 0.048356
Epoch 8800 | Loss: 0.045552
Epoch 9000 | Loss: 0.042973
Epoch 9200 | Loss: 0.040600
Epoch 9400 | Loss: 0.038415
Epoch 9600 | Loss: 0.036400
Epoch 9800 | Loss: 0.034540

MAE Predictions:
  [0.000000, 0.000000] -> 0.1695 (expected: 0.0000)
  [0.0000, 1.0000] -> 0.7970 (expected: 1.0000)
  [1.0000, 0.0000] -> 0.8171 (expected: 1.0000)
  [1.0000, 1.0000] -> 0.1670 (expected: 0.0000)

Process finished with exit code 0
*/

/*
with MAE:
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
=== MAE vs MSE Comparison ===

--- MAE Training ---
Epoch    0 | Loss: 0.502730
Epoch  200 | Loss: 0.492480
Epoch  400 | Loss: 0.490835
Epoch  600 | Loss: 0.492258
Epoch  800 | Loss: 0.493666
Epoch 1000 | Loss: 0.494742
Epoch 1200 | Loss: 0.495546
Epoch 1400 | Loss: 0.496156
Epoch 1600 | Loss: 0.496631
Epoch 1800 | Loss: 0.497008
Epoch 2000 | Loss: 0.497314
Epoch 2200 | Loss: 0.497566
Epoch 2400 | Loss: 0.497777
Epoch 2600 | Loss: 0.497956
Epoch 2800 | Loss: 0.498110
Epoch 3000 | Loss: 0.498243
Epoch 3200 | Loss: 0.498360
Epoch 3400 | Loss: 0.498462
Epoch 3600 | Loss: 0.498553
Epoch 3800 | Loss: 0.498635
Epoch 4000 | Loss: 0.498708
Epoch 4200 | Loss: 0.498774
Epoch 4400 | Loss: 0.498834
Epoch 4600 | Loss: 0.498888
Epoch 4800 | Loss: 0.498938
Epoch 5000 | Loss: 0.498984
Epoch 5200 | Loss: 0.499026
Epoch 5400 | Loss: 0.499065
Epoch 5600 | Loss: 0.499101
Epoch 5800 | Loss: 0.499134
Epoch 6000 | Loss: 0.499165
Epoch 6200 | Loss: 0.499194
Epoch 6400 | Loss: 0.499222
Epoch 6600 | Loss: 0.499247
Epoch 6800 | Loss: 0.499271
Epoch 7000 | Loss: 0.499294
Epoch 7200 | Loss: 0.499315
Epoch 7400 | Loss: 0.499335
Epoch 7600 | Loss: 0.499354
Epoch 7800 | Loss: 0.499372
Epoch 8000 | Loss: 0.499389
Epoch 8200 | Loss: 0.499405
Epoch 8400 | Loss: 0.499420
Epoch 8600 | Loss: 0.499435
Epoch 8800 | Loss: 0.499449
Epoch 9000 | Loss: 0.499462
Epoch 9200 | Loss: 0.499475
Epoch 9400 | Loss: 0.499487
Epoch 9600 | Loss: 0.499499
Epoch 9800 | Loss: 0.499510

MAE Predictions:
[0.000000, 0.000000] -> 0.9952 (expected: 0.0000)
[0.0000, 1.0000] -> 0.9981 (expected: 1.0000)
[1.0000, 0.0000] -> 0.9976 (expected: 1.0000)
[1.0000, 1.0000] -> 0.9986 (expected: 0.0000)

Process finished with exit code 0
*/