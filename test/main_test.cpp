//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    auto train_df = load(R"(..\test\archive\Student_Performance.csv)");
    auto test_df = load(R"(..\test\archive\test.csv)");

    train_df.Set("Performance Index");
    test_df.Set("Performance Index");

    train_df.drop("Extracurricular Activities");
    test_df.drop("Extracurricular Activities");

    train_df["Hours Studied"].scale();
    test_df["Hours Studied"].scale();
    train_df["Previous Scores"].scale();
    test_df["Previous Scores"].scale();
    train_df["Sleep Hours"].scale();
    test_df["Sleep Hours"].scale();
    train_df["Sample Question Papers Practiced"].scale();
    test_df["Sample Question Papers Practiced"].scale();
    train_df["Performance Index"].scale();
    test_df["Performance Index"].scale();

    auto [x_train, y_train] = train_df.split();
    auto [x_test, y_test] = test_df.split();

    net::Model model;

    model.add<nn::Dense>(4, 32);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(32, 16);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(16, 8);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(8, 1);

    model.compile<loss::MeanSquared, opt::Adam>();
    model.summary();

    model.fit(x_train, y_train, 1000, 100);

    auto pred = model.predict(x_test);

    for (size_t i = 0; i < (pred.len() / 10); ++i) {
        std::cout << "Predict: " << pred.get()[i] << " | " << "Target: " << y_test.get()[i] << std::endl;
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
Dense (4, 32)                 Yes
ReLU                          Yes
Dense (32, 16)                Yes
ReLU                          Yes
Dense (16, 8)                 Yes
ReLU                          Yes
Dense (8, 1)                  Yes
==================================================
Is compiled   : Yes
Loss Function : MSE
Optimizer     : Adam(0.001000)
Total Params  : 833
==================================================
Epoch 0     | Loss: 0.384230%
Epoch 100   | Loss: 0.017908%
Epoch 200   | Loss: 0.000806%
Epoch 300   | Loss: 0.000635%
Epoch 400   | Loss: 0.000592%
Epoch 500   | Loss: 0.000568%
Epoch 600   | Loss: 0.000555%
Epoch 700   | Loss: 0.000548%
Epoch 800   | Loss: 0.000543%
Epoch 900   | Loss: 0.000540%
Predict: 0.896104 | Target: 0.915663
Predict: 0.593127 | Target: 0.602410
Predict: 0.386574 | Target: 0.361446
Predict: 0.290214 | Target: 0.253012
Predict: 0.638126 | Target: 0.614458
Predict: 0.554311 | Target: 0.554217
Predict: 0.601561 | Target: 0.578313
Predict: 0.313906 | Target: 0.325301
Predict: 0.593992 | Target: 0.554217

Process finished with exit code 0
*/