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

TEST(DenseTest, DebugShape) {
    cortex::nn::Dense layer(4, 3, cortex::host);

    const auto weights = layer.getWeight();

    const auto s = weights[0].get().shape();

    std::cout << "shape size: " << s.size() << std::endl;
}

TEST(DenseTest, BackwardPass) {
    cortex::nn::Dense layer(2, 2, cortex::host);

    cortex::tensor input({1, 2}, cortex::host, true);
    input.rand();

    cortex::tensor output = layer.forward(input);
    output.sum().backward();

    const auto grads = layer.getGradient();

    EXPECT_NE(grads[0].get().mean(), 0.0f);
}

/*
 * output:
 * C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_G_TEST.exe --gtest_color=no
 * Testing started at 16:27 ...
 * Running main() from C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\_deps\googletest-src\googletest\src\gtest_main.cc
 * C:\software\Cpp\projects\CortexMind\test\g_test.cpp(41): error: Expected: (grads[0].get().mean()) != (0.0f), actual: 0 vs 0



 * Process finished with exit code 1
 */