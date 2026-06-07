//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;
/*
int main() {
    auto train_df = load(R"(..\test\archive\antenna_dataset.csv)");
    train_df.Set("Status");

    net::Model model;

    model.add<nn::Dense>(8, 64);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(64, 32);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(32, 16);
    model.add<nn::ReLU>();
    model.add<nn::Dropout>();
    model.add<nn::Dense>(16, 1);
    model.add<nn::Sigmoid>();

    model.compile<loss::BinaryCrossEntropy, opt::Adam, metric::Accuracy>();
    model.summary();

    auto[x, y] = train_df.split();

    model.fit(x, y, 100, 10);

    return 0;
}
*/
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe

==================================================
Model:
==================================================
Layer                         Mode
--------------------------------------------------
Dense(8, 64)                  Train
ReLU                          Train
Dense(64, 32)                 Train
ReLU                          Train
Dense(32, 16)                 Train
ReLU                          Train
Dropout(0.100000)             Train
Dense(16, 1)                  Train
Sigmoid                       Train
==================================================
Is compiled   : Yes
Loss Function : BCE(0.000000)
Optimizer     : Adam(0.001000)
Metric        : Accuracy
Total Params  : 3201
==================================================
Epoch 0     | Loss: 0.603262 | Accuracy: 0.829333
Epoch 10    | Loss: 0.415008 | Accuracy: 0.852667
Epoch 20    | Loss: 0.274494 | Accuracy: 0.925333
Epoch 30    | Loss: 0.186358 | Accuracy: 0.958000
Epoch 40    | Loss: 0.127733 | Accuracy: 0.990000
Epoch 50    | Loss: 0.088438 | Accuracy: 0.997333
Epoch 60    | Loss: 0.060636 | Accuracy: 0.998000
Epoch 70    | Loss: 0.043456 | Accuracy: 0.998667
Epoch 80    | Loss: 0.031501 | Accuracy: 1.000000
Epoch 90    | Loss: 0.023113 | Accuracy: 0.999333

Process finished with exit code 0
*/


int main() {
    auto train_df = load(R"(..\test\archive\antenna_dataset.csv)");
    train_df.one_hot("Fault_Type");
    train_df.Set({
    "Fault_Type_0.000000",
    "Fault_Type_1.000000",
    "Fault_Type_2.000000",
    "Fault_Type_3.000000",
    "Fault_Type_4.000000",
    "Fault_Type_5.000000"
    });

    net::Model model;

    model.add<nn::Dense>(8, 64);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(64, 32);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(32, 16);
    model.add<nn::ReLU>();
    model.add<nn::Dropout>();
    model.add<nn::Dense>(16, 6);

    model.compile<loss::CategoricalCrossEntropyWithLogit, opt::Adam, metric::Accuracy>();
    model.summary();

    auto[x, y] = train_df.split();

    model.fit(x, y, 250, 50);

    return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe

==================================================
Model:
==================================================
Layer                         Mode
--------------------------------------------------
Dense(8, 64)                  Train
ReLU                          Train
Dense(64, 32)                 Train
ReLU                          Train
Dense(32, 16)                 Train
ReLU                          Train
Dropout(0.100000)             Train
Dense(16, 6)                  Train
==================================================
Is compiled   : Yes
Loss Function : CCEWithLogit
Optimizer     : Adam(0.001000)
Metric        : Accuracy
Total Params  : 3286
==================================================
[ERROR] [CortexMind\framework\Engine\IX\TensorInit\init.cpp | 160] Storage is invalid

Process finished with exit code 1
*/