//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    const tensor x1({2, 2}, host, true);
    const tensor x2({2, 2}, host, true);

    x1.uniform();
    x2.uniform();

    const auto z = x1 * x2;

    std::cout << "Tensor 1:\n" << x1 << std::endl;
    std::cout << "Tensor 2:\n" << x2 << std::endl;
    std::cout << "Result:\n" << z << std::endl;

    z.sum().backward();

    std::cout << "Gradient 1:\n" << x1.grad() << std::endl;
    std::cout << "Gradient 2:\n" << x2.grad() << std::endl;

    return 0;
}

/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
Tensor 1:
[[0.902306, 0.8869],
 [0.223133, 0.0318382]]
Tensor 2:
[[0.482539, 0.128267],
 [0.636245, 0.813701]]
Result:
[[0.435398, 0.11376],
 [0.141967, 0.0259068]]
Gradient 1:
[[0.482539, 0.128267],
 [0.636245, 0.813701]]
Gradient 2:
[[0.902306, 0.8869],
 [0.223133, 0.0318382]]

Process finished with exit code 0
*/