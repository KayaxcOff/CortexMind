//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    auto train_df = load(R"(..\test\archive\student_train.csv)");
    auto test_df = load(R"(..\test\archive\student_test.csv)");

    train_df.label_encode("placement_status");
    test_df.label_encode("placement_status");

    train_df.Set("placement_status");
    test_df.Set("placement_status");

    train_df["study_hours"].scale();
    test_df["study_hours"].scale();
    train_df["attendance"].scale();
    test_df["attendance"].scale();
    train_df["sleep_hours"].scale();
    test_df["sleep_hours"].scale();
    train_df["internet_usage"].scale();
    test_df["internet_usage"].scale();
    train_df["assignments_completed"].scale();
    test_df["assignments_completed"].scale();
    train_df["previous_score"].scale();
    test_df["previous_score"].scale();
    train_df["exam_score"].scale();
    test_df["exam_score"].scale();

    auto [x_train, y_train] = train_df.split();
    auto [x_test, y_test] = test_df.split();

    net::Model model;

    model.add<nn::Dense>(7, 16);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(16, 32);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(32, 32);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(32, 16);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(16, 1);
    model.add<nn::Sigmoid>();

    model.compile<loss::BinaryCrossEntropy, opt::Adam>();
    model.summary();

    model.fit(x_train, y_train, 1000, 100);

    auto pred = model.predict(x_test);

    for (size_t i = 0; i < 10; i++) {
        std::cout << "Predict: " << pred.get()[i] << " Target: " << y_test.get()[i] << std::endl;
    }

    return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe

==================================================
Model:
==================================================
Layer                         Mode
--------------------------------------------------
Dense (7, 16)                 Train
ReLU                          Train
Dense (16, 32)                Train
ReLU                          Train
Dense (32, 32)                Train
ReLU                          Train
Dense (32, 16)                Train
ReLU                          Train
Dense (16, 1)                 Train
Sigmoid                       Train
==================================================
Is compiled   : Yes
Loss Function : BCE(0.000000)
Optimizer     : Adam(0.001000)
Total Params  : 2273
==================================================
Epoch 0     | Loss: 0.746220%
Epoch 100   | Loss: 0.278850%
Epoch 200   | Loss: 0.040181%
Epoch 300   | Loss: 0.015640%
Epoch 400   | Loss: 0.009944%
Epoch 500   | Loss: 0.007196%
Epoch 600   | Loss: 0.005592%
Epoch 700   | Loss: 0.004572%
Epoch 800   | Loss: 0.003850%
Epoch 900   | Loss: 0.003278%
Predict: 0.000000 Target: 0.000000
Predict: 0.000000 Target: 0.000000
Predict: 0.000000 Target: 0.000000
Predict: 0.000000 Target: 0.000000
Predict: 0.000000 Target: 0.000000
Predict: 0.000027 Target: 0.000000
Predict: 0.000000 Target: 0.000000
Predict: 0.000000 Target: 0.000000
Predict: 0.000000 Target: 0.000000
Predict: 0.000000 Target: 0.000000

Process finished with exit code 0
*/