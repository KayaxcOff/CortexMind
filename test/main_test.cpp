//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iomanip>
#include <ios>
#include <iostream>
#include <random>
#include <vector>

using namespace cortex;
/*
int main() {
    ds::TwoMoons df(100, 0.1);

    tensor X({df.N, 2}, df.X.data(), host);
    tensor Y({df.N, 1}, df.Y.data(), host);

    nn::Dense hidden(2, 16, host);
    nn::GeLUExact gelu_exact;
    nn::Dense output_layer(16, 1, host);
    nn::Sigmoid sigmoid;

    loss::MeanSquared mse;
    opt::StochasticGradient sgd(0.5f);

    auto params = hidden.getParameters();
    auto out_params = output_layer.getParameters();
    params.insert(params.end(), out_params.begin(), out_params.end());
    sgd.SetParams(params);

    constexpr int epochs = 10000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        tensor h  = hidden.forward(X);
        tensor h2 = gelu_exact.forward(h);
        tensor out = output_layer.forward(h2);
        tensor y_pred = sigmoid.forward(out);
        tensor l = mse.forward(y_pred, Y);

        sgd.zero_grad();
        l.backward();
        sgd.update();

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << std::setw(5) << epoch << " | Loss: " << std::fixed << std::setprecision(6) << l.get()[0] << std::endl;
        }
    }

    tensor h   = hidden.forward(X);
    tensor h2  = gelu_exact.forward(h);
    tensor out = output_layer.forward(h2);
    tensor pred = sigmoid.forward(out);

    for (int i = 0; i < 4; ++i) {
        std::cout << "  [" << std::fixed << std::setprecision(1) << df.X[i*2] << ", " << df.X[i*2+1] << "]" << " -> " << std::fixed << std::setprecision(4) << pred.get()[i] << " (expected: " << df.Y[i] << ")" << std::endl;
    }

    return 0;
}
*/
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
Epoch     0 | Loss: 0.258421
Epoch   200 | Loss: 0.086881
Epoch   400 | Loss: 0.084950
Epoch   600 | Loss: 0.079641
Epoch   800 | Loss: 0.075324
Epoch  1000 | Loss: 0.075247
Epoch  1200 | Loss: 0.070321
Epoch  1400 | Loss: 0.067812
Epoch  1600 | Loss: 0.060237
Epoch  1800 | Loss: 0.043324
Epoch  2000 | Loss: 0.025311
Epoch  2200 | Loss: 0.018470
Epoch  2400 | Loss: 0.010644
Epoch  2600 | Loss: 0.006775
Epoch  2800 | Loss: 0.005159
Epoch  3000 | Loss: 0.004077
Epoch  3200 | Loss: 0.003308
Epoch  3400 | Loss: 0.002746
Epoch  3600 | Loss: 0.002324
Epoch  3800 | Loss: 0.001999
Epoch  4000 | Loss: 0.001743
Epoch  4200 | Loss: 0.001538
Epoch  4400 | Loss: 0.001370
Epoch  4600 | Loss: 0.001232
Epoch  4800 | Loss: 0.001115
Epoch  5000 | Loss: 0.001016
Epoch  5200 | Loss: 0.000931
Epoch  5400 | Loss: 0.000858
Epoch  5600 | Loss: 0.000794
Epoch  5800 | Loss: 0.000737
Epoch  6000 | Loss: 0.000687
Epoch  6200 | Loss: 0.000643
Epoch  6400 | Loss: 0.000603
Epoch  6600 | Loss: 0.000568
Epoch  6800 | Loss: 0.000535
Epoch  7000 | Loss: 0.000506
Epoch  7200 | Loss: 0.000480
Epoch  7400 | Loss: 0.000456
Epoch  7600 | Loss: 0.000433
Epoch  7800 | Loss: 0.000413
Epoch  8000 | Loss: 0.000394
Epoch  8200 | Loss: 0.000377
Epoch  8400 | Loss: 0.000361
Epoch  8600 | Loss: 0.000346
Epoch  8800 | Loss: 0.000332
Epoch  9000 | Loss: 0.000319
Epoch  9200 | Loss: 0.000307
Epoch  9400 | Loss: 0.000296
Epoch  9600 | Loss: 0.000285
Epoch  9800 | Loss: 0.000275
  [-0.5, 0.7] -> 0.0037 (expected: 0.0000)
  [0.4, 0.6] -> 0.0138 (expected: 0.0000)
  [0.9, 0.5] -> 0.0004 (expected: 0.0000)
  [0.9, 0.7] -> 0.0000 (expected: 0.0000)

Process finished with exit code 0
*/
/*
int main() {
    ds::Spiral df(100, 0.1);

    tensor X({df.N, 2}, df.X.data(), host);
    tensor Y({df.N, 1}, df.Y.data(), host);

    nn::Dense hidden(2, 16, host);
    nn::GeLUExact gelu_exact;
    nn::Dense output_layer(16, 1, host);
    nn::Sigmoid sigmoid;

    loss::MeanSquared mse;
    opt::StochasticGradient sgd(0.5f);

    auto params = hidden.getParameters();
    auto out_params = output_layer.getParameters();
    params.insert(params.end(), out_params.begin(), out_params.end());
    sgd.SetParams(params);

    constexpr int epochs = 10000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        tensor h  = hidden.forward(X);
        tensor h2 = gelu_exact.forward(h);
        tensor out = output_layer.forward(h2);
        tensor y_pred = sigmoid.forward(out);
        tensor l = mse.forward(y_pred, Y);

        sgd.zero_grad();
        l.backward();
        sgd.update();

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << std::setw(5) << epoch << " | Loss: " << std::fixed << std::setprecision(6) << l.get()[0] << std::endl;
        }
    }

    tensor h   = hidden.forward(X);
    tensor h2  = gelu_exact.forward(h);
    tensor out = output_layer.forward(h2);
    tensor pred = sigmoid.forward(out);

    for (int i = 0; i < 4; ++i) {
        std::cout << "  [" << std::fixed << std::setprecision(1) << df.X[i*2] << ", " << df.X[i*2+1] << "]" << " -> " << std::fixed << std::setprecision(4) << pred.get()[i] << " (expected: " << df.Y[i] << ")" << std::endl;
    }

    return 0;
}
*/
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
Epoch     0 | Loss: 0.247976
Epoch   200 | Loss: 0.244156
Epoch   400 | Loss: 0.243694
Epoch   600 | Loss: 0.243357
Epoch   800 | Loss: 0.242974
Epoch  1000 | Loss: 0.242433
Epoch  1200 | Loss: 0.241636
Epoch  1400 | Loss: 0.240564
Epoch  1600 | Loss: 0.239242
Epoch  1800 | Loss: 0.237691
Epoch  2000 | Loss: 0.235980
Epoch  2200 | Loss: 0.234221
Epoch  2400 | Loss: 0.232519
Epoch  2600 | Loss: 0.230938
Epoch  2800 | Loss: 0.229461
Epoch  3000 | Loss: 0.228007
Epoch  3200 | Loss: 0.226477
Epoch  3400 | Loss: 0.224777
Epoch  3600 | Loss: 0.222799
Epoch  3800 | Loss: 0.220520
Epoch  4000 | Loss: 0.218326
Epoch  4200 | Loss: 0.216760
Epoch  4400 | Loss: 0.215813
Epoch  4600 | Loss: 0.215217
Epoch  4800 | Loss: 0.214793
Epoch  5000 | Loss: 0.214458
Epoch  5200 | Loss: 0.214171
Epoch  5400 | Loss: 0.213907
Epoch  5600 | Loss: 0.213649
Epoch  5800 | Loss: 0.213385
Epoch  6000 | Loss: 0.213113
Epoch  6200 | Loss: 0.212833
Epoch  6400 | Loss: 0.212541
Epoch  6600 | Loss: 0.212232
Epoch  6800 | Loss: 0.211897
Epoch  7000 | Loss: 0.211522
Epoch  7200 | Loss: 0.211083
Epoch  7400 | Loss: 0.210571
Epoch  7600 | Loss: 0.209990
Epoch  7800 | Loss: 0.209306
Epoch  8000 | Loss: 0.208506
Epoch  8200 | Loss: 0.207686
Epoch  8400 | Loss: 0.213041
Epoch  8600 | Loss: 0.207572
Epoch  8800 | Loss: 0.207083
Epoch  9000 | Loss: 0.206807
Epoch  9200 | Loss: 0.206685
Epoch  9400 | Loss: 0.206704
Epoch  9600 | Loss: 0.206875
Epoch  9800 | Loss: 0.207225
  [0.0, 0.1] -> 0.3337 (expected: 0.0000)
  [-0.0, 0.1] -> 0.2572 (expected: 0.0000)
  [0.0, -0.0] -> 0.4990 (expected: 0.0000)
  [-0.1, 0.2] -> 0.2023 (expected: 0.0000)

Process finished with exit code 0
*/

int main() {
    ds::TwoMoons df(100, 0.1);

    tensor X({df.N, 2}, df.X.data(), host);
    tensor Y({df.N, 1}, df.Y.data(), host);

    nn::Dense hidden(2, 16, host);
    nn::BatchNormalization bn({0}, 16, 0.1f, 1e-5f, host);  // ← Add
    nn::GeLUExact gelu_exact;
    nn::Dense output_layer(16, 1, host);
    nn::Sigmoid sigmoid;

    loss::MeanSquared mse;
    opt::StochasticGradient sgd(0.5f);

    auto params2 = hidden.getParameters();
    auto bn_params = bn.getParameters();
    auto out_params2 = output_layer.getParameters();

    auto params = hidden.getParameters();
    auto out_params = output_layer.getParameters();
    params.insert(params.end(), bn_params.begin(), bn_params.end());
    params.insert(params.end(), out_params.begin(), out_params.end());
    sgd.SetParams(params);

    constexpr int epochs = 10000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        tensor h  = hidden.forward(X);
        tensor h2 = bn.forward(h);          // ← BatchNorm
        tensor h3 = gelu_exact.forward(h2);
        tensor out = output_layer.forward(h3);
        tensor y_pred = sigmoid.forward(out);
        tensor l = mse.forward(y_pred, Y);

        sgd.zero_grad();
        l.backward();
        sgd.update();

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << std::setw(5) << epoch << " | Loss: " << std::fixed << std::setprecision(6) << l.get()[0] << std::endl;
        }
    }

    bn.EvalMode();

    tensor h  = hidden.forward(X);
    tensor h2 = bn.forward(h);          // ← BatchNorm
    tensor h3 = gelu_exact.forward(h2);
    tensor out = output_layer.forward(h3);
    tensor pred = sigmoid.forward(out);

    for (int i = 0; i < 4; ++i) {
        std::cout << "  [" << std::fixed << std::setprecision(1) << df.X[i*2] << ", " << df.X[i*2+1] << "]" << " -> " << std::fixed << std::setprecision(4) << pred.get()[i] << " (expected: " << df.Y[i] << ")" << std::endl;
    }

    return 0;
}
/*
It took so long, like more than 30 minute
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
Epoch     0 | Loss: 0.245013
Epoch   200 | Loss: 0.051378
Epoch   400 | Loss: 0.032729
Epoch   600 | Loss: 0.020316
Epoch   800 | Loss: 0.015957
Epoch  1000 | Loss: 0.013435
Epoch  1200 | Loss: 0.012094
Epoch  1400 | Loss: 0.011718
Epoch  1600 | Loss: 0.013569
Epoch  1800 | Loss: 0.010954
Epoch  2000 | Loss: 0.022091
Epoch  2200 | Loss: 0.023972

Process finished with exit code -1073741510 (0xC000013A: interrupted by Ctrl+C)

*/