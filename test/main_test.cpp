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