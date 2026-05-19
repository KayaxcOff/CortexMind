//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
        std::cout << "=== Scalar Operations Backward Test ===" << std::endl;

        tensor x({2, 2}, host, true);
        x.fill(3.0f);

        std::cout << "Input x = 3:\n" << x << std::endl;

        auto z1 = x + 2.0f;
        auto z2 = z1 * 5.0f;
        auto z3 = z2 - 1.0f;
        auto z4 = z3 / 2.0f;

        std::cout << "\nz = ((x + 2) * 5 - 1) / 2 = " << z4.get()[0] << std::endl;

        z4.sum().backward();

        std::cout << "\nGradient of x:" << std::endl;
        std::cout << x.grad() << std::endl;


        return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
=== Scalar Operations Backward Test ===
Input x = 3:
[[3, 3],
 [3, 3]]

z = ((x + 2) * 5 - 1) / 2 = 12

Gradient of x:
[[8.5, 8.5],
 [8.5, 8.5]]

Process finished with exit code 0
*/