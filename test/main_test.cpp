//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

int main() {
    auto train_df = load(R"(..\test\archive\train.csv)");
    train_df.Set("price_range");

    train_df["battery_power"].scale();
    train_df["int_memory"].scale();
    train_df["mobile_wt"].scale();
    train_df["n_cores"].scale();
    train_df["pc"].scale();
    train_df["px_height"].scale();
    train_df["px_width"].scale();
    train_df["ram"].scale();
    train_df["sc_h"].scale();
    train_df["sc_w"].scale();
    train_df["talk_time"].scale();

    net::Model model;

    model.add<nn::Dense>(20, 64);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(64, 32);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(32, 16);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(16, 4);
    model.add<nn::Softmax>();

    model.compile<loss::CategoricalCrossEntropy, opt::Adam, metric::Accuracy>();
    model.summary();

    auto[x, y] = train_df.split();

    model.fit(x, y, 10);

    return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe

==================================================
Model:
==================================================
Layer                         Mode
--------------------------------------------------
Dense(20, 64)                 Train
ReLU                          Train
Dense(64, 32)                 Train
ReLU                          Train
Dense(32, 16)                 Train
ReLU                          Train
Dense(16, 4)                  Train
Softmax                       Train
==================================================
Is compiled   : Yes
Loss Function : CCE(0.000000)
Optimizer     : Adam(0.001000)
Metric        : Accuracy
Total Params  : 4020
==================================================
Epoch 0     | Loss: 0.017890 | Accuracy: 0.250000
Epoch 100   | Loss: 0.017890 | Accuracy: 0.250000
Epoch 200   | Loss: 0.017890 | Accuracy: 0.250000
Epoch 300   | Loss: 0.017890 | Accuracy: 0.250000
Epoch 400   | Loss: 0.017890 | Accuracy: 0.250000
Epoch 500   | Loss: 0.017890 | Accuracy: 0.250000
Epoch 600   | Loss: 0.017890 | Accuracy: 0.250000
Epoch 700   | Loss: 0.017890 | Accuracy: 0.250000
Epoch 800   | Loss: 0.017890 | Accuracy: 0.250000
Epoch 900   | Loss: 0.017890 | Accuracy: 0.250000
Epoch 1000  | Loss: 0.017890 | Accuracy: 0.250000
Epoch 1100  | Loss: 0.017890 | Accuracy: 0.250000
Epoch 1200  | Loss: 0.017890 | Accuracy: 0.250000
Epoch 1300  | Loss: 0.017890 | Accuracy: 0.250000
Epoch 1400  | Loss: 0.017890 | Accuracy: 0.250000

Process finished with exit code 0
*/