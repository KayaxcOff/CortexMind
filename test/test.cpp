//
// Created by muham on 19.12.2025.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex::tools;
using namespace cortex;

int main() {

    const tensor output = TextVec::to_tensor(R"(C:\software\Cpp\projects\CortexMind\test\sample.csv)");
    output.print();

    return 0;
}