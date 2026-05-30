//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {

    auto image = utils::VisionModule::load(R"(..\test\test01.jpg)");
    image = image.unsqueeze(0);

    image = image.div(255.0f);

    nn::Conv2D l1(3,16,3,3,2,2,1,1);
    nn::ReLU l2;

    tensor output = l1.forward(image);
    output = l2.forward(output);

    std::cout << "Max: " << output.max() << std::endl;
    std::cout << "Min: " << output.min() << std::endl;
    std::cout << "Mean: " << output.mean() << std::endl;
    std::cout << "Variance: " << output.variance() << std::endl;

    return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
Max: 2.87184
Min: 0
Mean: 0.477196
Variance: 0.35541

Process finished with exit code 0
*/