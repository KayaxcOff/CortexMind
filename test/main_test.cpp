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
    model.add<nn::Dense>(16, 1);

    model.compile<loss::MeanSquared, opt::Adam>();
    auto y_true = tensor({1, 1});
    y_true.fill(1.0f);
    model.fit(image, y_true, 5);

    return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
[ERROR] [CortexMind\framework\Tensor\tensor.cpp | 362] reshape requires a contiguous tensor

Process finished with exit code 1
*/