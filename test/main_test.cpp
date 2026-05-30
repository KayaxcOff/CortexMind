//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    auto test_df = load(R"(..\test\archive\student_test.csv)");

    test_df.label_encode("placement_status");

    test_df.Set("placement_status");

    test_df["study_hours"].scale();
    test_df["attendance"].scale();
    test_df["sleep_hours"].scale();
    test_df["internet_usage"].scale();
    test_df["assignments_completed"].scale();
    test_df["previous_score"].scale();
    test_df["exam_score"].scale();

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
    model.load(R"(..\test\model)");

    auto pred = model.predict(x_test);

    for (size_t i = 0; i < 20; i++) {
        auto pred_val = pred.get()[i];
        std::cout << "Predict: " << (pred_val > 0.5f ? 1 : 0) << "\n";
        std::cout << "Expected: " << y_test.get()[i] << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 1
Expected: 1

Predict: 0
Expected: 0

Predict: 0
Expected: 0

Predict: 0
Expected: 0


Process finished with exit code 0

*/