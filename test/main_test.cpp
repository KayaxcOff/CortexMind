//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    auto train_df = load(R"(C:\software\Cpp\projects\CortexMind\test\test.csv)");

    train_df.drop("A2");

    train_df["A3"].scale();

    train_df.Set("A3");
    auto [x, y] = train_df.split();
    std::cout << "X:\n" << x << std::endl;
    std::cout << "Y:\n" << y << std::endl;

    return 0;
}

/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
No NAN

Process finished with exit code -1073741819 (0xC0000005)
*/