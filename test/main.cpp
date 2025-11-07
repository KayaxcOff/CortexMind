//
// Created by muham on 5.11.2025.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>
#include <vector>

using namespace cortex;

void test1() {
    model::Model model;

    const std::vector<math::MindVector> X = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
    };
    const std::vector<math::MindVector> Y = {
        {0.0}, {1.0}, {1.0}, {0.0}
    };
    const math::MindVector test = {1.0, 0.0};

    model.add<layer::Dense>(2, 10);
    model.add<layer::Dense>(10, 1);
    model.compile<loss::MeanSquared, optim::StochasticGradient>(0.01);

    try {
        constexpr size_t epochs = 2;
        model.fit(X, Y, epochs);
    } catch (const std::exception & e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
    }

    const auto result_test = model.predict(test);

    result_test.print("Result of Test");
}

void test2() {
    model::Model model;

    tools::MindTokenizer tokenizer;

    const std::string input  = "Cortex Mind is fun!  ";
    const std::string target = "Cortex Mind is cool! ";

    const std::vector<math::MindVector> X = tokenizer.tokenize(input);
    const std::vector<math::MindVector> Y = tokenizer.tokenize(target);

    model.add<layer::Dense>(X[0].size(), 10);
    model.add<layer::Dense>(10, Y[0].size());

    model.compile<loss::MeanSquared, optim::StochasticGradient>(0.001);

    try {
        constexpr size_t epochs = 10;
        model.fit(X, Y, epochs);
    } catch (const std::exception & e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
    }

    std::cout << std::endl;

    const math::MindVector& test = X[5];
    const auto result_test = model.predict(test);
    result_test.print("Result of Test");
}

void test3() {
    model::Model model;
    tools::MindImage image;

    const std::string path = R"(C:\software\Cpp\projects\CortexMind\test\test.png)";

    const std::vector<math::MindVector> X = image.transformPixels(path);
    std::vector<math::MindVector> Y;
    for (size_t i = 0; i < X.size(); ++i) {
        Y.emplace_back(10, 0.5);
    }

    model.add<layer::Dense>(X[0].size(), 20);
    model.add<layer::Dense>(20, Y[0].size());
    model.compile<loss::MeanSquared, optim::StochasticGradient>(0.001);
    try {
        constexpr size_t epochs = 5;
        model.fit({X[0]}, {Y[0]}, epochs);
    } catch (const std::exception & e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
    }

    const auto test = model.predict(X[0]);

    test.print("Result of Test");
}

int main() {
    test1();

    return 0;
}