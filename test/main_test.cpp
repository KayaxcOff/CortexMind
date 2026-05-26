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

    for (size_t i = 123; i < 126; ++i) {
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
Epoch 0     | Loss: 0.696275%
Epoch 500   | Loss: 0.626950%
Epoch 1000  | Loss: 0.410603%
Epoch 1500  | Loss: 0.248763%
Epoch 2000  | Loss: 0.181638%
Epoch 2500  | Loss: 0.140614%
Epoch 3000  | Loss: 0.115951%
Epoch 3500  | Loss: 0.100175%
Epoch 4000  | Loss: 0.089573%
Epoch 4500  | Loss: 0.077082%
Epoch 5000  | Loss: 0.064277%
Epoch 5500  | Loss: 0.055680%
Epoch 6000  | Loss: 0.051013%
Epoch 6500  | Loss: 0.047020%
Epoch 7000  | Loss: 0.040149%
Epoch 7500  | Loss: 0.037445%
Epoch 8000  | Loss: 0.035479%
Epoch 8500  | Loss: 0.033978%
Epoch 9000  | Loss: 0.032805%
Epoch 9500  | Loss: 0.031848%
Epoch 10000 | Loss: 0.031056%
Epoch 10500 | Loss: 0.030389%
Epoch 11000 | Loss: 0.028411%
Epoch 11500 | Loss: 0.025510%
Epoch 12000 | Loss: 0.024428%
Epoch 12500 | Loss: 0.023737%
Epoch 13000 | Loss: 0.016548%
Epoch 13500 | Loss: 0.015577%
Epoch 14000 | Loss: 0.015118%
Epoch 14500 | Loss: 0.014812%
Epoch 15000 | Loss: 0.014573%
Epoch 15500 | Loss: 0.014381%
Epoch 16000 | Loss: 0.014226%
Epoch 16500 | Loss: 0.014093%
Epoch 17000 | Loss: 0.011984%
Epoch 17500 | Loss: 0.005682%
Epoch 18000 | Loss: 0.003115%
Epoch 18500 | Loss: 0.001979%
Epoch 19000 | Loss: 0.001372%
Epoch 19500 | Loss: 0.001010%
 [1.1, -0.2] -> 0.0000 (expected: 0.0000)
 [0.9, -0.2] -> 0.0000 (expected: 0.0000)
 [0.1, -0.1] -> 1.0000 (expected: 1.0000)

Process finished with exit code 0

*/