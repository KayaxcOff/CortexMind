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
    model.add<nn::Dropout>();
    model.add<nn::Dense>(16, 1);
    model.add<nn::SigmoidFast>();

    model.compile<loss::BinaryCrossEntropy, opt::StochasticGradient>(0.5f);

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
Dropout(0.100000)             Yes
Dense (16, 1)                 Yes
SigmoidFast                   Yes
==================================================
Is compiled   : Yes
Loss Function : BCE(0.000000)
Optimizer     : SGD(0.500000)
Total Params  : 65
==================================================
Epoch 0     | Loss: 0.431928
Epoch 100   | Loss: 0.026754
Epoch 200   | Loss: 0.020910
Epoch 300   | Loss: 0.004886
Epoch 400   | Loss: 0.008654
Epoch 500   | Loss: 0.021927
Epoch 600   | Loss: 0.026674
Epoch 700   | Loss: 0.023654
Epoch 800   | Loss: 0.032258
Epoch 900   | Loss: 0.002122
Epoch 1000  | Loss: 0.027028
Epoch 1100  | Loss: 0.013127
Epoch 1200  | Loss: 0.002063
Epoch 1300  | Loss: 0.029269
Epoch 1400  | Loss: 0.001770
  [-0.7, 0.2] -> 0.0000 (expected: 0.0000)
  [-0.6, 0.9] -> 0.0000 (expected: 0.0000)
  [0.7, 0.8] -> 0.0000 (expected: 0.0000)
  [0.0, -0.6] -> 0.0000 (expected: 0.0000)

Process finished with exit code 0
*/