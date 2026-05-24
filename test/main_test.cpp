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
    ds::CircleDataset df(1000, 0.1f);

    tensor X({df.N, 2}, df.X.data(), host);
    tensor Y({df.N, 1}, df.Y.data(), host);

    nn::Dense hidden(2, 16, host);
    nn::Tanh tanh;
    nn::Dense output_layer(16, 1, host);
    nn::Sigmoid sigmoid;

    loss::MeanSquared mse;
    opt::StochasticGradient sgd(0.1f);

    auto params = hidden.getParameters();
    auto out_params = output_layer.getParameters();
    params.insert(params.end(), out_params.begin(), out_params.end());
    sgd.SetParams(params);

    constexpr int epochs = 100000;
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
                  << df.X[i*2] << ", " << df.X[i*2+1] << "]"
                  << " -> " << std::fixed << std::setprecision(4) << pred.get()[i]
                  << " (expected: " << df.Y[i] << ")" << std::endl;
    }

    return 0;
}
*/
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
Epoch    0 | Loss: 0.300326
... (Too many line(like between 400-500), so I didn't want to add them)
Epoch 99800 | Loss: 0.044714
  [0.8, -0.6] -> 0.0297 (expected: 0.0000)
  [-0.5, -0.9] -> 0.0077 (expected: 0.0000)
  [1.0, -0.1] -> 0.0036 (expected: 0.0000)
  [0.9, 0.3] -> 0.0002 (expected: 0.0000)

Process finished with exit code 0
*/
/*
int main() {
    ds::CircleDataset df(1000, 0.1f);

    tensor X({df.N, 2}, df.X.data(), host);
    tensor Y({df.N, 1}, df.Y.data(), host);

    nn::Dense hidden(2, 16, host);
    nn::GeLU gelu;
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
        tensor h2 = gelu.forward(h);
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
    tensor h2  = gelu.forward(h);
    tensor out = output_layer.forward(h2);
    tensor pred = sigmoid.forward(out);

    for (int i = 0; i < 4; ++i) {
        std::cout << "  [" << std::fixed << std::setprecision(1)
                  << df.X[i*2] << ", " << df.X[i*2+1] << "]"
                  << " -> " << std::fixed << std::setprecision(4) << pred.get()[i]
                  << " (expected: " << df.Y[i] << ")" << std::endl;
    }

    return 0;
}
*/
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
Epoch    0 | Loss: 0.241964
Epoch  200 | Loss: 0.128748
Epoch  400 | Loss: 0.081564
Epoch  600 | Loss: 0.056986
Epoch  800 | Loss: 0.047710
Epoch 1000 | Loss: 0.043253
Epoch 1200 | Loss: 0.040607
Epoch 1400 | Loss: 0.038807
Epoch 1600 | Loss: 0.037479
Epoch 1800 | Loss: 0.036455
Epoch 2000 | Loss: 0.035645
Epoch 2200 | Loss: 0.034996
Epoch 2400 | Loss: 0.034470
Epoch 2600 | Loss: 0.034042
Epoch 2800 | Loss: 0.033690
Epoch 3000 | Loss: 0.033397
Epoch 3200 | Loss: 0.033153
Epoch 3400 | Loss: 0.032947
Epoch 3600 | Loss: 0.032771
Epoch 3800 | Loss: 0.032620
Epoch 4000 | Loss: 0.032489
Epoch 4200 | Loss: 0.032375
Epoch 4400 | Loss: 0.032274
Epoch 4600 | Loss: 0.032185
Epoch 4800 | Loss: 0.032106
Epoch 5000 | Loss: 0.032035
Epoch 5200 | Loss: 0.031971
Epoch 5400 | Loss: 0.031913
Epoch 5600 | Loss: 0.031860
Epoch 5800 | Loss: 0.031812
Epoch 6000 | Loss: 0.031768
Epoch 6200 | Loss: 0.031727
Epoch 6400 | Loss: 0.031689
Epoch 6600 | Loss: 0.031655
Epoch 6800 | Loss: 0.031622
Epoch 7000 | Loss: 0.031593
Epoch 7200 | Loss: 0.031565
Epoch 7400 | Loss: 0.031539
Epoch 7600 | Loss: 0.031515
Epoch 7800 | Loss: 0.031492
Epoch 8000 | Loss: 0.031471
Epoch 8200 | Loss: 0.031451
Epoch 8400 | Loss: 0.031432
Epoch 8600 | Loss: 0.031414
Epoch 8800 | Loss: 0.031398
Epoch 9000 | Loss: 0.031382
Epoch 9200 | Loss: 0.031367
Epoch 9400 | Loss: 0.031353
Epoch 9600 | Loss: 0.031340
Epoch 9800 | Loss: 0.031327
  [0.6, -0.5] -> 0.0090 (expected: 0.0000)
  [0.7, -0.3] -> 0.0254 (expected: 0.0000)
  [0.4, -0.8] -> 0.0029 (expected: 0.0000)
  [-0.0, 0.0] -> 0.9993 (expected: 1.0000)

Process finished with exit code 0
*/
/*
int main() {
    ds::CircleDataset df(100, 0.1f);

    tensor X({df.N, 2}, df.X.data(), host);
    tensor Y({df.N, 1}, df.Y.data(), host);

    nn::Dense hidden(2, 16, host);
    nn::LeakyReLU leaky_relu(0.01f);
    nn::Dense output_layer(16, 1, host);
    nn::Sigmoid sigmoid;

    loss::MeanSquared mse;
    opt::StochasticGradient sgd(0.5f);

    auto params = hidden.getParameters();
    auto out_params = output_layer.getParameters();
    params.insert(params.end(), out_params.begin(), out_params.end());
    sgd.SetParams(params);

    constexpr int epochs = 100000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        tensor h  = hidden.forward(X);
        tensor h2 = leaky_relu.forward(h);
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
    tensor h2  = leaky_relu.forward(h);
    tensor out = output_layer.forward(h2);
    tensor pred = sigmoid.forward(out);

    for (int i = 0; i < 4; ++i) {
        std::cout << "  [" << std::fixed << std::setprecision(1)
                  << df.X[i*2] << ", " << df.X[i*2+1] << "]"
                  << " -> " << std::fixed << std::setprecision(4) << pred.get()[i]
                  << " (expected: " << df.Y[i] << ")" << std::endl;
    }

    return 0;
}
*/
/*
Epoch    0 | Loss: 0.216305
... (too many line)
Epoch 99800 | Loss: 0.005576
[-0.8, -0.2] -> 0.0000 (expected: 0.0000)
[-0.8, -0.2] -> 0.0000 (expected: 0.0000)
[-0.2, -0.0] -> 0.9998 (expected: 1.0000)
[-0.8, 0.1] -> 0.0000 (expected: 0.0000)
*/
/*
int main() {
    ds::CircleDataset df(100, 0.1f);

    tensor X({df.N, 2}, df.X.data(), host);
    tensor Y({df.N, 1}, df.Y.data(), host);

    nn::Dense hidden(2, 16, host);
    nn::GeLUExact gelu_exact;  // ← Test
    nn::Dense output_layer(16, 1, host);
    nn::Sigmoid sigmoid;

    loss::MeanSquared mse;
    opt::StochasticGradient sgd(0.5f);

    auto params = hidden.getParameters();
    auto out_params = output_layer.getParameters();
    params.insert(params.end(), out_params.begin(), out_params.end());
    sgd.SetParams(params);

    constexpr int epochs = 100000;
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
            std::cout << "Epoch " << std::setw(5) << epoch
                     << " | Loss: " << std::fixed << std::setprecision(6)
                     << l.get()[0] << std::endl;
        }
    }

    tensor h   = hidden.forward(X);
    tensor h2  = gelu_exact.forward(h);
    tensor out = output_layer.forward(h2);
    tensor pred = sigmoid.forward(out);

    for (int i = 0; i < 4; ++i) {
        std::cout << "  [" << std::fixed << std::setprecision(1)
                  << df.X[i*2] << ", " << df.X[i*2+1] << "]"
                  << " -> " << std::fixed << std::setprecision(4) << pred.get()[i]
                  << " (expected: " << df.Y[i] << ")" << std::endl;
    }

    return 0;
}
*/
/*
Epoch     0 | Loss: 0.242213
(Too many line.)
Epoch 99800 | Loss: 0.001111
  [-0.8, -1.1] -> 0.0000 (expected: 0.0000)
  [0.2, -0.1] -> 1.0000 (expected: 1.0000)
  [-0.6, 0.9] -> 0.0000 (expected: 0.0000)
  [-1.0, -0.5] -> 0.0000 (expected: 0.0000)
*/

int main() {
    ds::CircleDataset df(100, 0.1f);

    tensor X({df.N, 2}, df.X.data(), host);
    tensor Y({df.N, 1}, df.Y.data(), host);

    nn::Dense hidden(2, 16, host);
    nn::SiLU silu;  // ← Test
    nn::Dense output_layer(16, 1, host);
    nn::Sigmoid sigmoid;

    loss::MeanSquared mse;
    opt::StochasticGradient sgd(0.5f);

    auto params = hidden.getParameters();
    auto out_params = output_layer.getParameters();
    params.insert(params.end(), out_params.begin(), out_params.end());
    sgd.SetParams(params);

    constexpr int epochs = 100000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        tensor h  = hidden.forward(X);
        tensor h2 = silu.forward(h);  // ← SiLU
        tensor out = output_layer.forward(h2);
        tensor y_pred = sigmoid.forward(out);
        tensor l = mse.forward(y_pred, Y);

        sgd.zero_grad();
        l.backward();
        sgd.update();

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << std::setw(5) << epoch
                     << " | Loss: " << std::fixed << std::setprecision(6)
                     << l.get()[0] << std::endl;
        }
    }

    tensor h   = hidden.forward(X);
    tensor h2  = silu.forward(h);
    tensor out = output_layer.forward(h2);
    tensor pred = sigmoid.forward(out);

    for (int i = 0; i < 4; ++i) {
        std::cout << "  [" << std::fixed << std::setprecision(1)
                  << df.X[i*2] << ", " << df.X[i*2+1] << "]"
                  << " -> " << std::fixed << std::setprecision(4) << pred.get()[i]
                  << " (expected: " << df.Y[i] << ")" << std::endl;
    }

    return 0;
}
/*
Epoch     0 | Loss: 0.253057
Epoch   200 | Loss: 0.146658
Epoch   400 | Loss: 0.134579
Epoch   600 | Loss: 0.112326
Epoch 20600 | Loss: 0.013433
Epoch 20800 | Loss: 0.154732
Epoch 21000 | Loss: 0.189982
Epoch 21200 | Loss: 0.189993
Epoch 21400 | Loss: 0.189996
Epoch 21600 | Loss: 0.189997
Epoch 21800 | Loss: 0.189998
Epoch 22000 | Loss: 0.189998
Epoch 22200 | Loss: 0.189999
Epoch 22400 | Loss: 0.189999
Epoch 22600 | Loss: 0.189999
Epoch 22800 | Loss: 0.189999
Epoch 23000 | Loss: 0.189999
Epoch 23200 | Loss: 0.189999
Epoch 23400 | Loss: 0.189999
Epoch 23600 | Loss: 0.189999
Epoch 23800 | Loss: 0.190000
Epoch 24000 | Loss: 0.190000
Epoch 24200 | Loss: 0.190000
Epoch 24400 | Loss: 0.190000
Epoch 99000 | Loss: 0.190000
Epoch 99200 | Loss: 0.190000
Epoch 99400 | Loss: 0.190000
Epoch 99600 | Loss: 0.190000
Epoch 99800 | Loss: 0.190000
[0.7, -0.3] -> 0.0000 (expected: 0.0000)
[0.4, 0.5] -> 0.0000 (expected: 0.0000)
[0.1, -0.9] -> 0.0000 (expected: 0.0000)
[-0.2, 0.3] -> 0.0000 (expected: 1.0000)
*/