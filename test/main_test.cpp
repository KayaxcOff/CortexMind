//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    const tensor x1({2, 2}, host, true);
    const tensor x2({2, 2}, host, true);
    const tensor x3({2, 2}, host, true);

    x1.uniform();
    x2.uniform();
    x3.uniform();

    std::cout << "Tensor 1:\n" << x1 << std::endl;
    std::cout << "Tensor 2:\n" << x2 << std::endl;
    std::cout << "Tensor 3:\n" << x3 << std::endl;

    auto z1 = x1 + x2;
    auto z2 = z1 - x3;

    std::cout << "Result:\n" << z2 << std::endl;

    z2.sum().backward();

    std::cout << "Gradient 1:\n" << x1.grad() << std::endl;
    std::cout << "Gradient 2:\n" << x2.grad() << std::endl;
    std::cout << "Gradient 3:\n" << x3.grad() << std::endl;

    return 0;
}

/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
Tensor 1:
[[0.0806152, 0.834729],
 [0.0522925, 0.53503]]
Tensor 2:
[[0.285536, 0.514128],
 [0.336753, 0.24319]]
Tensor 3:
[[0.266811, 0.818149],
 [0.560359, 0.305607]]
Result:
[[0.0993395, 0.530709],
 [-0.171314, 0.472612]]
Gradient 1:
[[1, 1],
 [1, 1]]
Gradient 2:
[[1, 1],
 [1, 1]]
Gradient 3:
[[-1, -1],
 [-1, -1]]

Process finished with exit code 0
*/
/*
#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    std::cout << "=== MatMul Backward Test ===" << std::endl;

    // (2,3) @ (3,2) = (2,2)
    const std::vector<float32> data_a = {1, 2, 3, 4, 5, 6};
    const std::vector<float32> data_b = {7, 8, 9, 10, 11, 12};

    tensor a({2, 3}, data_a.data(), host, true);
    tensor b({3, 2}, data_b.data(), host, true);

    std::cout << "\nA:\n" << a << std::endl;
    std::cout << "\nB:\n" << b << std::endl;

    const auto result = a.matmul(b);

    std::cout << "\nA @ B:\n" << result << std::endl;

    const auto s = result.sum();
    s.backward();

    std::cout << "\nGradient A:\n" << a.grad() << std::endl;
    std::cout << "\nGradient B:\n" << b.grad() << std::endl;

    return 0;
}
*/
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
=== MatMul Backward Test ===

A:
[[1, 2, 3],
 [4, 5, 6]]

B:
[[7, 8],
 [9, 10],
 [11, 12]]

A @ B:
[[58, 64],
 [139, 154]]

Gradient A:
[[17, 19, 21],
 [17, 19, 21]]

Gradient B:
[[3, 3],
 [7, 7],
 [11, 11]]

Process finished with exit code 0
*/