//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h> // from _deps/googletest-src

TEST(DenseTest, Initialization) {
    cortex::nn::Dense layer(4, 3, cortex::host);

    const auto weights = layer.getWeight();

    EXPECT_EQ(weights[0].get().shape()[0], 4);
    EXPECT_EQ(weights[0].get().shape()[1], 3);

    EXPECT_EQ(weights[1].get().shape()[0], 1);
    EXPECT_EQ(weights[1].get().shape()[1], 3);
}

/*
 * output:
Testing started at 17:23 ...
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_G_TEST.exe --gtest_color=no
Running main() from C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\_deps\googletest-src\googletest\src\gtest_main.cc
unknown file: error: SEH exception with code 0xc0000005 thrown in the test body.
Stack trace:




Process finished with exit code 1
 */

TEST(DenseTest, DebugShape) {
    cortex::nn::Dense layer(4, 3, cortex::host);

    const auto weights = layer.getWeight();

    const auto s = weights[0].get().shape();

    std::cout << "shape size: " << s.size() << std::endl;
}
/*
Testing started at 17:23 ...
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_G_TEST.exe --gtest_color=no
Running main() from C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\_deps\googletest-src\googletest\src\gtest_main.cc
unknown file: error: SEH exception with code 0xc0000005 thrown in the test body.
Stack trace:




Process finished with exit code 1

 */