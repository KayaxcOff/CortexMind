//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iomanip>

using namespace cortex;

int main() {
    ds::Spiral s(250, 0.1f);

    const tensor X({s.N, 2}, s.X.data(), host);
    const tensor Y({s.N, 1}, s.Y.data(), host);

    net::Model model;

    model.add<nn::Dense>(2, 16);
    model.add<nn::LeakyReLU>();

    model.add<nn::Dense>(16, 1);
    model.add<nn::Sigmoid>();

    model.compile<loss::MeanSquared, opt::Adam>(0.0001f);
    model.summary();

    model.fit(X, Y, 1000, 16);

    model.eval();
    auto pred = model.predict(X);

    for (size_t i = 0; i < 6; ++i) {
        std::cout << " [" << std::fixed << std::setprecision(1) << s.X[i * 2] << ", " << s.X[i * 2 + 1] << "]" << " -> " << std::fixed << std::setprecision(4) << pred.get()[i] << " (expected: " << s.Y[i] << ")" << std::endl;
    }

    size_t correct = 0;
    for (size_t i = 0; i < s.N; ++i) {
        float expected = s.Y[i];
        float predicted_class = (pred.get()[i] >= 0.5f) ? 1.0f : 0.0f;

        if (predicted_class == expected) {
            correct++;
        }
    }
    float accuracy = static_cast<float>(correct) / static_cast<float>(s.N) * 100.0f;
    std::cout << "==================================================" << std::endl;
    std::cout << "Model Test Accuracy: %" << std::fixed << std::setprecision(2) << accuracy << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}

/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe

==================================================
Model:
==================================================
Layer                         Trainable
--------------------------------------------------
Dense (2, 16)                 Yes
LeakyReLU(0.010000)           Yes
Dense (16, 1)                 Yes
Sigmoid                       Yes
==================================================
Is compiled   : Yes
Loss Function : MSE
Optimizer     : Adam(0.000100)
Total Params  : 65
==================================================
Epoch 0     | Loss: 0.252330
Epoch 100   | Loss: 0.087955
Epoch 200   | Loss: 0.020255
Epoch 300   | Loss: 0.005691
Epoch 400   | Loss: 0.001906
Epoch 500   | Loss: 0.000705
Epoch 600   | Loss: 0.000276
Epoch 700   | Loss: 0.000112
Epoch 800   | Loss: 0.000046
Epoch 900   | Loss: 0.000019
 [0.0, 0.0] -> 0.0034 (expected: 0.0000)
 [0.1, -0.0] -> 0.0031 (expected: 0.0000)
 [0.1, 0.1] -> 0.0023 (expected: 0.0000)
 [0.2, 0.1] -> 0.0020 (expected: 0.0000)
 [0.0, 0.2] -> 0.0017 (expected: 0.0000)
 [-0.0, 0.1] -> 0.0028 (expected: 0.0000)
==================================================
Model Test Accuracy: %50.00
==================================================

Process finished with exit code 0

*/