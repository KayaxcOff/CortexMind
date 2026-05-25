//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iomanip>

using namespace cortex;

int main() {
    ds::Circle df(100, 0.1);

    const tensor X({df.N, 2}, df.X.data(), host);
    const tensor Y({df.N, 1}, df.Y.data(), host);

    net::Model model;

    model.add<nn::Dense>(2, 16);
    model.add<nn::GeLUExact>();
    model.add<nn::Dense>(16, 1);
    model.add<nn::SigmoidFast>();

    model.compile<loss::MeanSquared, opt::StochasticGradient>(0.5f);

    model.summary();

    model.fit(X, Y, 1500, 16);

    model.eval();
    auto pred = model.predict(X);

    for (size_t i = 0; i < 4; ++i) {
        std::cout << "  [" << std::fixed << std::setprecision(1) << df.X[i * 2] << ", " << df.X[i * 2 + 1] << "]" << " -> " << std::fixed << std::setprecision(4) << pred.get()[i] << " (expected: " << df.Y[i] << ")" << std::endl;
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
Dense (2, 16)                 Yes
GeLUExact                     Yes
Dense (16, 1)                 Yes
SigmoidFast                   Yes
==================================================
Loss Function : MSE
Optimizer     : SGD(0.500000)
Total Params  : 65
==================================================
Epoch 0     | Loss: 0.205050
Epoch 100   | Loss: 0.045407
Epoch 200   | Loss: 0.010855
Epoch 300   | Loss: 0.003798
Epoch 400   | Loss: 0.001822
Epoch 500   | Loss: 0.001094
Epoch 600   | Loss: 0.000746
Epoch 700   | Loss: 0.000551
Epoch 800   | Loss: 0.000429
Epoch 900   | Loss: 0.000348
Epoch 1000  | Loss: 0.000290
Epoch 1100  | Loss: 0.000247
Epoch 1200  | Loss: 0.000215
Epoch 1300  | Loss: 0.000189
Epoch 1400  | Loss: 0.000168
  [-0.3, -0.5] -> 0.9739 (expected: 1.0000)
  [0.8, 0.7] -> 0.0000 (expected: 0.0000)
  [0.5, 0.1] -> 0.0000 (expected: 0.0000)
  [-0.6, 0.4] -> 0.0000 (expected: 0.0000)

Process finished with exit code 0
*/