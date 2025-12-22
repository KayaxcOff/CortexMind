//
// Created by muham on 19.12.2025.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;
using namespace cortex::_fw;

int main() {
    const tensor output = tools::ImageNetLoader::imagenet(R"(C:\software\Cpp\projects\CortexMind\test\indir.jpg)");

    std::cout << "Output's batch   : " << output.batch() << std::endl;
    std::cout << "Output's channel : " << output.channel() << std::endl;
    std::cout << "Output's height  : " << output.height() << std::endl;
    std::cout << "Output's width   : " << output.width() << std::endl;

    const int in_channel  = output.channel();
    constexpr int out_channel = 8;
    constexpr int kH = 3;
    constexpr int kW = 3;

    MindKernel kernel(in_channel, out_channel, kH, kW);

    tensor conv_result = kernel.apply(output);

    std::cout << "\nConvolution Result:" << std::endl;
    std::cout << "Batch   : " << conv_result.batch() << std::endl;
    std::cout << "Channel : " << conv_result.channel() << std::endl;
    std::cout << "Height  : " << conv_result.height() << std::endl;
    std::cout << "Width   : " << conv_result.width() << std::endl;

    std::cout << "\nFirst row of first channel after convolution:" << std::endl;
    for (int i = 0; i < conv_result.width(); ++i) {
        std::cout << conv_result.at(0, 0, 0, i) << " ";
    }
    std::cout << std::endl;

    return 0;
}
