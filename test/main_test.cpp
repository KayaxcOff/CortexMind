//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

int main() {

    auto image = utils::VisionModule::load(R"(..\test\test01.jpg)");
    image = image.unsqueeze(0);

    image = image.div(255.0f);

    net::Model model;

    model.add<nn::Conv2D>(3,16,3,3,8,8,1,1);
    model.add<nn::ReLU>();
    model.add<nn::GlobalAveragePool2D>();
    model.add<nn::Dense>(16, 8);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(8, 8);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(8, 16);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(16, 16);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(16, 8);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(8, 16);
    model.add<nn::ReLU>();
    model.add<nn::Dropout>();
    model.add<nn::Dense>(16, 1);

    model.compile<loss::MeanSquared, opt::Adam>(1e-6f);
    model.summary();

    auto y_true = tensor({1, 1});
    y_true.fill(1.0f);
    model.fit(image, y_true, 5);

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
Dense(16, 8)                  Train
ReLU                          Train
Dense(8, 8)                   Train
ReLU                          Train
Dense(8, 16)                  Train
ReLU                          Train
Dense(16, 16)                 Train
ReLU                          Train
Dense(16, 8)                  Train
ReLU                          Train
Dense(8, 16)                  Train
ReLU                          Train
Dropout(0.100000)             Train
Dense(16, 1)                  Train
==================================================
Is compiled   : Yes
Loss Function : MSE
Optimizer     : Adam(0.000001)
Total Params  : 1369
==================================================
Epoch 0     | Loss: 1.096755%
Epoch 1     | Loss: inf%
Epoch 2     | Loss: -nan(ind)%
Epoch 3     | Loss: -nan(ind)%
Epoch 4     | Loss: -nan(ind)%

Process finished with exit code 0

*/