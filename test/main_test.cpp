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

struct CircleDataset {
    std::vector<float> X;
    std::vector<float> Y;

    int N;

    explicit CircleDataset(const int n, const float noise = 0.1f) : N(n) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution dist(-1.0f, 1.0f);
        std::normal_distribution gauss(0.0f, noise);

        this->X.reserve(n * 2);
        this->Y.reserve(n);

        for (int i = 0; i < n; i++) {
            float x = dist(gen);
            float y = dist(gen);

            const float r = std::sqrt(x*x + y*y);

            float label = (r < 0.5f) ? 1.0f : 0.0f;

            // noise
            x += gauss(gen);
            y += gauss(gen);

            this->X.push_back(x);
            this->X.push_back(y);
            this->Y.push_back(label);
        }
    }
};

int main() {
    CircleDataset df(1000, 0.1f);

    tensor X({4, 2}, df.X.data(), host);
    tensor Y({4, 1}, df.Y.data(), host);

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
                  << df.X[i*2] << ", " << df.X[i*2+1] << "]"
                  << " -> " << std::fixed << std::setprecision(4) << pred.get()[i]
                  << " (expected: " << df.Y[i] << ")" << std::endl;
    }

    return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
Epoch    0 | Loss: 0.182190
Epoch  200 | Loss: 0.122991
Epoch  400 | Loss: 0.080459
Epoch  600 | Loss: 0.052098
Epoch  800 | Loss: 0.034207
Epoch 1000 | Loss: 0.023192
Epoch 1200 | Loss: 0.016457
Epoch 1400 | Loss: 0.012228
Epoch 1600 | Loss: 0.009460
Epoch 1800 | Loss: 0.007567
Epoch 2000 | Loss: 0.006221
Epoch 2200 | Loss: 0.005229
Epoch 2400 | Loss: 0.004476
Epoch 2600 | Loss: 0.003890
Epoch 2800 | Loss: 0.003424
Epoch 3000 | Loss: 0.003047
Epoch 3200 | Loss: 0.002737
Epoch 3400 | Loss: 0.002477
Epoch 3600 | Loss: 0.002258
Epoch 3800 | Loss: 0.002071
Epoch 4000 | Loss: 0.001910
Epoch 4200 | Loss: 0.001770
Epoch 4400 | Loss: 0.001647
Epoch 4600 | Loss: 0.001538
Epoch 4800 | Loss: 0.001442
Epoch 5000 | Loss: 0.001356
Epoch 5200 | Loss: 0.001279
Epoch 5400 | Loss: 0.001209
Epoch 5600 | Loss: 0.001146
Epoch 5800 | Loss: 0.001088
Epoch 6000 | Loss: 0.001036
Epoch 6200 | Loss: 0.000988
Epoch 6400 | Loss: 0.000944
Epoch 6600 | Loss: 0.000903
Epoch 6800 | Loss: 0.000866
Epoch 7000 | Loss: 0.000831
Epoch 7200 | Loss: 0.000798
Epoch 7400 | Loss: 0.000768
Epoch 7600 | Loss: 0.000740
Epoch 7800 | Loss: 0.000714
Epoch 8000 | Loss: 0.000689
Epoch 8200 | Loss: 0.000666
Epoch 8400 | Loss: 0.000644
Epoch 8600 | Loss: 0.000624
Epoch 8800 | Loss: 0.000605
Epoch 9000 | Loss: 0.000586
Epoch 9200 | Loss: 0.000569
Epoch 9400 | Loss: 0.000553
Epoch 9600 | Loss: 0.000537
Epoch 9800 | Loss: 0.000523
  [-0.6, -1.2] -> 0.0001 (expected: 0.0000)
  [-0.6, 0.3] -> 0.0126 (expected: 0.0000)
  [-0.1, -0.1] -> 0.9678 (expected: 1.0000)
  [-0.3, -0.3] -> 0.0289 (expected: 0.0000)

Process finished with exit code 0
*/