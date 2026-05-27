//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    auto train_df = load(R"(C:\software\Cpp\projects\CortexMind\test\test.csv)");
    train_df.Set("A3");
    train_df["A3"].scale();

    auto [x, y] = train_df.split();
    std::cout << "X:\n" << x << std::endl;
    std::cout << "Y:\n" << y << std::endl;

    return 0;
}

/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
X:
[[1, 2, 4, 5],
 [6, 7, 9, 10],
 [11, 12, 14, 15],
 [16, 17, 19, 20],
 [21, 22, 24, 25]]
Y:
[[0],
 [0.25],
 [0.5],
 [0.75],
 [1]]

Process finished with exit code 0

-----------------
test.csv
A1,A2,A3,A4,A5,
1,2,3,4,5,
6,7,8,9,10,
11,12,13,14,15,
16,17,18,19,20,
21,22,23,24,25
*/