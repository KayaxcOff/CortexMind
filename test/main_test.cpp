//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    const tensor x1({3, 4}, cuda);
    const tensor x2({1, 4}, cuda);

    x1.rand();
    x2.rand();

    std::cout << "X1:\n" << x1 << std::endl;
    std::cout << "X2:\n" << x2 << std::endl;
    std::cout << "X1 + X2:\n" << x1 + x2 << std::endl;

    return 0;
}

/**
X1:
[[0.0700209, 0.160146, 0.420496, 0.576371],
 [0.157703, 0.401926, 0.507685, 0.108753],
 [0.966218, 0.53876, 0.160213, 0.95207]]

X2:
[[0.673353, 0.439168, 0.351157, 0.0857413]]

X1 + X2:
[[0.743374, 0.599314, 0.771653, 0.662112],
 [0.831056, 0.841094, 0.858841, 0.194495],
 [1.63957, 0.977928, 0.51137, 1.03781]]


Process finished with exit code 0

 */