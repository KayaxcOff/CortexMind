//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

int main() {

    auto image = utils::VisionModule::load(R"(..\test\test01.jpg)");
    image = utils::VisionModule::normalize(image);
    image = image.unsqueeze(0);

    net::Model model;

    model.add<nn::Conv2D>(3, 16, 3, 3, 8, 8, 1, 1);
    model.add<nn::ReLU>();
    model.add<nn::GlobalAveragePool2D>();
    model.add<nn::Dense>(16, 1);

    model.compile<loss::MeanSquared, opt::Adam>(0.01f);
    model.summary();

    auto y_true = tensor({1, 1});
    y_true.fill(1.0f);

    model.fit(image, y_true, 100, 10);


    return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe

==================================================
Model:
==================================================
Layer                         Mode
--------------------------------------------------
Conv2D(16)                    Train
ReLU                          Train
GlobalAveragePooling2D        Train
Dense(16, 1)                  Train
==================================================
Is compiled   : Yes
Loss Function : MSE
Optimizer     : Adam(0.010000)
Total Params  : 465
==================================================
Epoch 0     | Loss: 1.932275%
Epoch 10    | Loss: 0.219304%
Epoch 20    | Loss: 0.000404%
Epoch 30    | Loss: 0.022602%
Epoch 40    | Loss: 0.002741%
Epoch 50    | Loss: 0.000184%
Epoch 60    | Loss: 0.000402%
Epoch 70    | Loss: 0.000036%
Epoch 80    | Loss: 0.000002%
Epoch 90    | Loss: 0.000005%

Process finished with exit code 0

*/