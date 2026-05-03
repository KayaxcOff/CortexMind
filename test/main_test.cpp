//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    const tensor x1({3, 4});
    const tensor x2({1, 4});

    x1.rand();
    x2.rand();

    std::cout << "X1:\n" << x1 << std::endl;
    std::cout << "X2:\n" << x2 << std::endl;
    std::cout << "X1 + X2:\n" << x1 + x2 << std::endl;

    return 0;
}

/**
 * C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_TEST.exe
 * X1:
 * [[0.676883, 0.416247, 0.754076, 0.305609],
 *  [0.128713, 0.0192522, 0.750098, 0.661509],
 *  [0.145962, 0.0625744, 0.757287, 0.0532617]]

 * X2:
 * [[0.368206, 0.475875, 0.318265, 0.151358]]

 * X1 + X2:
 * [[-4.31602e+08, -4.31602e+08, -4.31602e+08, -4.31602e+08],
 * [-4.31602e+08, -4.31602e+08, -4.31602e+08, -4.31602e+08],
 * [-4.31602e+08, -4.31602e+08, -4.31602e+08, -4.31602e+08]]


 * Process finished with exit code 0

 */