//
// Created by muham on 19.12.2025.
//

#include <CortexMind/cortexmind.hpp>
#include <vector>

using namespace cortex;

int main() {
    tensor x(3, 1, 1, 2);

    x.uniform_rand();

    tensor y(3, 1, 1, 1);

    y.uniform_rand();

    std::vector batchX = {x};
    std::vector batchY = {y};

    tin::LinearRegression linear(2);

    linear.fit(batchX, batchY);

    const tensor predict = linear.predict(x);

    log("True tensor:");
    y.print();

    std::cout << std::endl;

    log("Predicted tensor:");
    predict.print();

    return 0;
}