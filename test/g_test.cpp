//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h>

using namespace cortex;

TEST(SumDimTest, LastDimCPU) {
    const std::vector<float32> data = {1,2,3,4, 5,6,7,8};
    const tensor t({2, 4}, data.data(), host, true);

    tensor result = t.sum({1}, true);

    ASSERT_EQ(result.shape()[0], 2);
    ASSERT_EQ(result.shape()[1], 1);
    EXPECT_NEAR(result.get()[0], 10.0f, 1e-4f);  // 1+2+3+4
    EXPECT_NEAR(result.get()[1], 26.0f, 1e-4f);  // 5+6+7+8
}

TEST(SumDimTest, FirstDimCPU) {
    const std::vector<float32> data = {1,2,3,4, 5,6,7,8};
    const tensor t({2, 4}, data.data(), host);

    tensor result = t.sum({0}, true);

    ASSERT_EQ(result.shape()[0], 1);
    ASSERT_EQ(result.shape()[1], 4);
    EXPECT_NEAR(result.get()[0], 6.0f,  1e-4f);  // 1+5
    EXPECT_NEAR(result.get()[1], 8.0f,  1e-4f);  // 2+6
    EXPECT_NEAR(result.get()[2], 10.0f, 1e-4f);  // 3+7
    EXPECT_NEAR(result.get()[3], 12.0f, 1e-4f);  // 4+8
}

TEST(SumDimTest, BackwardLastDim) {
    const std::vector<float32> data = {1,2,3,4, 5,6,7,8};
    tensor t({2, 4}, data.data(), host, true);

    const tensor result = t.sum({1}, true);   // (2,1)
    result.sum().backward();

    for (size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(t.grad().get()[i], 1.0f, 1e-4f);
    }
}

// Conv2D Test
TEST(Conv2DBackwardTest, BiasGradValue) {
    nn::Conv2D conv(1, 1, 3, 3);

    auto params = conv.getParameters();
    params[0].get().fill(1.0f);
    params[1].get().zero();

    tensor input({1, 1, 5, 5}, host, true);
    input.fill(1.0f);

    tensor output = conv.forward(input);  // shape (1,1,3,3)
    tensor loss = output.sum();
    loss.backward();

    auto grads = conv.getGradients();

    EXPECT_NEAR(grads[1].get().get()[0], 9.0f, 1e-3f);
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_G_TEST.exe --gtest_color=no
Testing started at 15:21 ...
Running main() from C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\_deps\googletest-src\googletest\src\gtest_main.cc
C:\software\Cpp\projects\CortexMind\test\g_test.cpp(135): error: The difference between grads[1].get().get()[0] and 9.0f is 9, which exceeds 1e-3f, where
grads[1].get().get()[0] evaluates to 18,
9.0f evaluates to 9, and
1e-3f evaluates to 0.0010000000474974513.



Process finished with exit code 1
*/
