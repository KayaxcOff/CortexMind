//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    auto train_df = load(R"(..\test\archive\train.csv)");
    auto test_df = load(R"(..\test\archive\test.csv)");

    train_df.Set("label");
    test_df.Set("label");

    auto[x_train, y_train] = train_df.split();
    auto[x_test, y_test] = test_df.split();

    net::Model model;

    model.add<nn::Dense>(4, 8);
    model.add<nn::Tanh>();
    model.add<nn::Dense>(8, 1);

    model.compile<loss::MeanSquared, opt::Adam>(0.01f);
    model.summary();
    model.fit(x_train, y_train, 5500, 500);

    auto pred = model.predict(x_test);

    for (size_t i = 0; i < pred.len(); ++i) {
        std::cout << "Model's prediction: " << pred.get()[i] << " | " << "Expected: " << y_test.get()[i] << std::endl;
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
Dense (4, 8)                  Yes
Tanh                          Yes
Dense (8, 1)                  Yes
==================================================
Is compiled   : Yes
Loss Function : MSE
Optimizer     : Adam(0.010000)
Total Params  : 49
==================================================
Epoch 0     | Loss: 5.132080%
Epoch 500   | Loss: 0.021289%
Epoch 1000  | Loss: 0.019647%
Epoch 1500  | Loss: 0.018079%
Epoch 2000  | Loss: 0.015896%
Epoch 2500  | Loss: 0.013604%
Epoch 3000  | Loss: 0.010163%
Epoch 3500  | Loss: 0.007085%
Epoch 4000  | Loss: 0.004808%
Epoch 4500  | Loss: 0.002974%
Epoch 5000  | Loss: 0.002312%
Model's prediction: -0.005140 | Expected: 0.000000
Model's prediction: 0.188028 | Expected: 0.000000
Model's prediction: 0.035691 | Expected: 0.000000
Model's prediction: 1.003301 | Expected: 1.000000
Model's prediction: 1.298586 | Expected: 1.000000
Model's prediction: 0.979590 | Expected: 1.000000
Model's prediction: 2.051482 | Expected: 2.000000
Model's prediction: 1.751830 | Expected: 2.000000
Model's prediction: 2.012960 | Expected: 2.000000
Model's prediction: 1.873987 | Expected: 2.000000

Process finished with exit code 0
*/