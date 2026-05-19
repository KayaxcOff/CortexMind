//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
     std::cout << "=== abs, neg, sign Backward Test ===" << std::endl;

     tensor x({2, 2}, host, true);
     x.fill(2.0f);
     x.get()[1] = -3.0f;  // Second element negative

     std::cout << "Input x:\n" << x << std::endl;

     auto y_abs = x.abs();
     auto y_neg = x.neg();

     std::cout << "\nabs(x):\n" << y_abs << std::endl;
     std::cout << "neg(x):\n" << y_neg << std::endl;

     // Test backward: L = sum(abs(x)) + sum(neg(x))
     auto result = y_abs + y_neg;
     result.sum().backward();

     std::cout << "\nGradient of x (expected: abs gradient + neg gradient):" << std::endl;
     std::cout << x.grad() << std::endl;

     // Manual calculation:
     // For x[0] = 2:
     //   ∂abs/∂x = sign(2) = 1
     //   ∂neg/∂x = -1
     //   Total = 1 + (-1) = 0
     // For x[1] = -3:
     //   ∂abs/∂x = sign(-3) = -1
     //   ∂neg/∂x = -1
     //   Total = -1 + (-1) = -2

     std::cout << "\nExpected gradient: [[0, 0], [-2, -2]] (approximately)" << std::endl;

     return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
=== abs, neg, sign Backward Test ===
Input x:
[[2, -3],
 [2, 2]]

abs(x):
[[2, 3],
 [2, 2]]
neg(x):
[[-2, -3],
 [-2, -2]]

Gradient of x (expected: abs gradient + neg gradient):
[[0, -2],
 [0, 0]]

Expected gradient: [[0, 0], [-2, -2]] (approximately)

Process finished with exit code 0
*/

/*
#include <CortexMind/cortexmind.hpp>
#include <iostream>
#include <cmath>

using namespace cortex;

int main() {
    std::cout << "=== Unary Operations Backward Test ===" << std::endl;

    tensor x({2, 2}, host, true);
    x.fill(2.0f);

    std::cout << "Input x = 2:\n" << x << std::endl;

    auto y_sqrt = x.sqrt();
    auto y_exp = x.exp();
    auto y_log = x.log();
    auto y_pow = x.pow(3.0f);
    auto y_sin = x.sin();
    auto y_cos = x.cos();

    std::cout << "\nsqrt(2) = " << y_sqrt.get()[0] << " (expected: " << std::sqrt(2.0f) << ")" << std::endl;
    std::cout << "exp(2) = " << y_exp.get()[0] << " (expected: " << std::exp(2.0f) << ")" << std::endl;
    std::cout << "log(2) = " << y_log.get()[0] << " (expected: " << std::log(2.0f) << ")" << std::endl;
    std::cout << "pow(2, 3) = " << y_pow.get()[0] << " (expected: 8)" << std::endl;
    std::cout << "sin(2) = " << y_sin.get()[0] << " (expected: " << std::sin(2.0f) << ")" << std::endl;
    std::cout << "cos(2) = " << y_cos.get()[0] << " (expected: " << std::cos(2.0f) << ")" << std::endl;

    // Test backward
    auto result = y_sqrt + y_exp + y_log + y_pow + y_sin + y_cos;
    result.sum().backward();

    std::cout << "\nGradient of x:" << std::endl;
    std::cout << x.grad() << std::endl;

    // Manual calculation for x = 2:
    // ∂sqrt/∂x = 1/(2*sqrt(2)) ≈ 0.3536
    // ∂exp/∂x = exp(2) ≈ 7.389
    // ∂log/∂x = 1/2 = 0.5
    // ∂pow/∂x = 3*2^2 = 12
    // ∂sin/∂x = cos(2) ≈ -0.416
    // ∂cos/∂x = -sin(2) ≈ -0.909
    // Total ≈ 19.456

    float32 expected_grad = 1.0f / (2.0f * std::sqrt(2.0f)) +
                          std::exp(2.0f) +
                          0.5f +
                          3.0f * 2.0f * 2.0f +
                          std::cos(2.0f) +
                          (-std::sin(2.0f));

    std::cout << "\nExpected gradient: " << expected_grad << std::endl;

    return 0;
}
*/
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
=== Unary Operations Backward Test ===
Input x = 2:
[[2, 2],
 [2, 2]]

sqrt(2) = 1.41421 (expected: 1.41421)
exp(2) = 7.38906 (expected: 7.38906)
log(2) = 0.693147 (expected: 0.693147)
pow(2, 3) = 8 (expected: 8)
sin(2) = 0.909297 (expected: 0.909297)
cos(2) = -0.416147 (expected: -0.416147)

Gradient of x:
[[-0.909297, -0.909297],
 [-0.909297, -0.909297]]

Expected gradient: 18.9172

Process finished with exit code 0
*/
/*
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
*/
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