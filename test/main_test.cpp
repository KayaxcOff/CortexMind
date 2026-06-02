//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

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

/*
int main() {
    auto train_df = load(R"(..\test\archive\antenna_dataset.csv)");
    train_df.one_hot("Fault_Type");
    train_df.Set("Fault_Type_1.000000");
    train_df.Set("Fault_Type_2.000000");
    train_df.Set("Fault_Type_3.000000");
    train_df.Set("Fault_Type_4.000000");
    train_df.Set("Fault_Type_5.000000");
    train_df.head();

    net::Model model;

    model.add<nn::Dense>(13, 64);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(64, 32);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(32, 16);
    model.add<nn::ReLU>();
    model.add<nn::Dropout>();
    model.add<nn::Dense>(16, 6);
    model.add<nn::Softmax>();

    model.compile<loss::CategoricalCrossEntropy, opt::Adam, metric::Accuracy>();
    model.summary();

    auto[x, y] = train_df.split();

    model.fit(x, y, 100, 10);

    return 0;
}

C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
S11 (dB)        VSWR    Gain (dBi)      Eff (%) BW (MHz)        Z_Real (Ohms)   Z_Imag (Ohms)   Status  Fault_Type_1.000
000     Fault_Type_4.000000     Fault_Type_2.000000     Fault_Type_0.000000     Fault_Type_5.000000     Fault_Type_3.000
000
0.0544662       -0.631356       0.422205        0.361451        0.904843        -0.0381401      0.519633        1
1       0       0       0       0       0
0.11523 -0.329512       0.397559        0.109627        0.698747        -0.3049 -0.689372       1       0       1
0       0       0       0
1.07153 0.836072        -1.27017        -1.14081        -1.23855        -1.47513        2.46161 1       0       0
1       0       0       0
0.961577        2.69357 -2.46552        -1.1712 -1.1149 -1.19532        1.66092 1       0       0       1       0
0       0
0.89792 0.654965        -0.921019       -0.915035       -1.32099        -1.19491        2.28713 1       0       0
1       0       0       0

==================================================
Model:
==================================================
Layer                         Mode
--------------------------------------------------
Dense(13, 64)                 Train
ReLU                          Train
Dense(64, 32)                 Train
ReLU                          Train
Dense(32, 16)                 Train
ReLU                          Train
Dropout(0.100000)             Train
Dense(16, 6)                  Train
Softmax                       Train
==================================================
Is compiled   : Yes
Loss Function : CCE(0.000000)
Optimizer     : Adam(0.001000)
Metric        : Accuracy
Total Params  : 3606
==================================================
Epoch 0     | Loss: 1.795672 | Accuracy: 0.833333
Epoch 10    | Loss: 16.118093 | Accuracy: 0.833333
Epoch 20    | Loss: 16.118093 | Accuracy: 0.833333
Epoch 30    | Loss: 16.118093 | Accuracy: 0.833333
Epoch 40    | Loss: 16.118093 | Accuracy: 0.833333
Epoch 50    | Loss: 16.118093 | Accuracy: 0.833333
Epoch 60    | Loss: 16.118093 | Accuracy: 0.833333
Epoch 70    | Loss: 16.118093 | Accuracy: 0.833333
Epoch 80    | Loss: 16.118093 | Accuracy: 0.833333
Epoch 90    | Loss: 16.118093 | Accuracy: 0.833333

Process finished with exit code 0

*/