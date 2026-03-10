//
// Created by muham on 21.02.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

int main() {
    tensor x = utils::VisionModule::load(R"(C:\software\Cpp\projects\CortexMind\test\archive\nagumo.jpg)");
    x.print_shape();

    return cortex::exit;
}