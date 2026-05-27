//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    auto train_df = load(R"(C:\software\Cpp\projects\CortexMind\test\test.csv)");

    train_df.drop("A2");

    if (train_df.is_nan()) {
        std::cout << "NAN" << std::endl;
    } else {
        std::cout << "No NAN" << std::endl;
    }

    train_df.Set("A3");
    auto [x, y] = train_df.split(1.0f);
    std::cout << x << "\n" << y << std::endl;

    return 0;
}

/*
No NAN
[ERROR] [CortexMind\framework\Tensor\tensor.cpp | 504] slice: invalid start 4
*/