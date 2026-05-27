//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iomanip>
#include <iostream>

using namespace cortex;

int main() {
    ds::Spiral s(250, 0.1f);

    const tensor X({s.N, 2}, s.X.data(), host);
    const tensor Y({s.N, 1}, s.Y.data(), host);

    net::Model model;

    model.add<nn::Dense>(2, 64);
    model.add<nn::Tanh>();

    model.add<nn::Dense>(64, 64);
    model.add<nn::Tanh>();

    model.add<nn::Dense>(64, 1);
    model.add<nn::Sigmoid>();

    model.compile<loss::BinaryCrossEntropy, opt::Adam>(0.001f);

    model.summary();

    model.fit(X, Y, 20000, 500);

    model.eval();
    auto pred = model.predict(X);

    for (size_t i = 123; i < 128; ++i) {
        std::cout << " [" << std::fixed << std::setprecision(1) << s.X[i * 2] << ", " << s.X[i * 2 + 1] << "]" << " -> " << std::fixed << std::setprecision(4) << pred.get()[i] << " (expected: " << s.Y[i] << ")" << std::endl;
    }

    return 0;
}

/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe

==================================================
Model:
==================================================
Layer                         Trainable
--------------------------------------------------
Dense (2, 64)                 Yes
Tanh                          Yes
Dense (64, 64)                Yes
Tanh                          Yes
Dense (64, 1)                 Yes
Sigmoid                       Yes
==================================================
Is compiled   : Yes
Loss Function : BCE(0.000000)
Optimizer     : Adam(0.001000)
Total Params  : 4417
==================================================
Epoch 0     | Loss: 0.688307%
Epoch 500   | Loss: 0.619313%
Epoch 1000  | Loss: 0.344934%
Epoch 1500  | Loss: 0.205641%
Epoch 2000  | Loss: 0.130343%
Epoch 2500  | Loss: 0.091893%
Epoch 3000  | Loss: 0.068547%
Epoch 3500  | Loss: 0.056063%
Epoch 4000  | Loss: 0.048978%
Epoch 4500  | Loss: 0.043269%
Epoch 5000  | Loss: 0.040104%
Epoch 5500  | Loss: 0.038060%
Epoch 6000  | Loss: 0.036675%
Epoch 6500  | Loss: 0.035721%
Epoch 7000  | Loss: 0.035012%
Epoch 7500  | Loss: 0.034469%
Epoch 8000  | Loss: 0.034067%
Epoch 8500  | Loss: 0.033765%
Epoch 9000  | Loss: 0.033518%
Epoch 9500  | Loss: 0.033337%
Epoch 10000 | Loss: 0.033197%
Epoch 10500 | Loss: 0.033079%
Epoch 11000 | Loss: 0.032991%
Epoch 11500 | Loss: 0.032919%
Epoch 12000 | Loss: 0.032863%
Epoch 12500 | Loss: 0.032810%
Epoch 13000 | Loss: 0.028583%
Epoch 13500 | Loss: 0.028205%
Epoch 14000 | Loss: 0.028046%
Epoch 14500 | Loss: 0.027963%
Epoch 15000 | Loss: 0.027909%
Epoch 15500 | Loss: 0.027870%
Epoch 16000 | Loss: 0.027843%
Epoch 16500 | Loss: 0.027824%
Epoch 17000 | Loss: 0.027809%
Epoch 17500 | Loss: 0.027798%
Epoch 18000 | Loss: 0.027789%
Epoch 18500 | Loss: 0.027782%
Epoch 19000 | Loss: 0.027777%
Epoch 19500 | Loss: 0.027772%
 [1.0, -0.3] -> 0.0000 (expected: 0.0000)
 [1.1, -0.1] -> 0.0000 (expected: 0.0000)
 [-0.1, -0.1] -> 1.0000 (expected: 1.0000)
 [-0.0, 0.1] -> 1.0000 (expected: 1.0000)
 [-0.1, 0.0] -> 1.0000 (expected: 1.0000)

Process finished with exit code 0
*/