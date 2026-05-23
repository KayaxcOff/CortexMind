//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h>

using namespace cortex;

TEST(Conv2DTest, OutputShape) {
    // input: (1, 1, 4, 4), kernel: (1, 1, 3, 3), stride=1, pad=0
    // output: (1, 1, 2, 2)
    nn::Conv2D conv(1, 1, 3, 3);

    tensor input({1, 1, 4, 4}, host, false);
    input.fill(1.0f);

    tensor output = conv.forward(input);

    ASSERT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 1);  // N
    EXPECT_EQ(output.shape()[1], 1);  // oC
    EXPECT_EQ(output.shape()[2], 2);  // oH
    EXPECT_EQ(output.shape()[3], 2);  // oW
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_G_TEST.exe --gtest_color=no
Testing started at 09:43 ...
Running main() from C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\_deps\googletest-src\googletest\src\gtest_main.cc
unknown file: error: SEH exception with code 0xc0000005 thrown in the test body.
Stack trace:




Process finished with exit code 3
*/

TEST(Conv2DTest, OutputShapeWithPadding) {
    // input: (1, 1, 4, 4), kernel: (1, 1, 3, 3), stride=1, pad=1
    // output: (1, 1, 4, 4) — same padding
    nn::Conv2D conv(1, 1, 3, 3, 1, 1, 1, 1);

    tensor input({1, 1, 4, 4}, host, false);
    input.fill(1.0f);

    tensor output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 4);
}

TEST(Conv2DTest, OutputShapeWithStride) {
    // input: (1, 1, 6, 6), kernel: (1, 1, 3, 3), stride=2, pad=0
    // output: (1, 1, 2, 2)
    nn::Conv2D conv(1, 1, 3, 3, 2, 2, 0, 0);

    tensor input({1, 1, 6, 6}, host, false);
    input.fill(1.0f);

    tensor output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 2);
    EXPECT_EQ(output.shape()[3], 2);
}

TEST(Conv2DTest, KnownWeightForward) {
    // identity kernel — weight=1, bias=0
    // input tüm 1'ler, 3x3 kernel → her output = 9
    nn::Conv2D conv(1, 1, 3, 3);

    // Weight'i 1'e set et
    auto params = conv.getParameters();
    params[0].get().fill(1.0f);  // weight
    params[1].get().zero();       // bias

    const std::vector<float32> input_data(1*1*5*5, 1.0f);
    tensor input({1, 1, 5, 5}, input_data.data(), host, false);

    tensor output = conv.forward(input);

    // output shape: (1, 1, 3, 3)
    EXPECT_EQ(output.shape()[2], 3);
    EXPECT_EQ(output.shape()[3], 3);

    // Her output = sum of 3x3 patch of 1s = 9
    for (size_t i = 0; i < output.len(); ++i) {
        EXPECT_NEAR(output.get()[i], 9.0f, 1e-4f);
    }
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_G_TEST.exe --gtest_color=no
Testing started at 09:44 ...
Running main() from C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\_deps\googletest-src\googletest\src\gtest_main.cc
unknown file: error: SEH exception with code 0xc0000005 thrown in the test body.
Stack trace:




Process finished with exit code 3
*/

TEST(Conv2DTest, MultiChannelShape) {
    // input: (2, 3, 8, 8), kernel: (16, 3, 3, 3)
    // output: (2, 16, 6, 6)
    nn::Conv2D conv(3, 16, 3, 3);

    tensor input({2, 3, 8, 8}, host, false);
    input.uniform(0.0f, 1.0f);

    tensor output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 6);
    EXPECT_EQ(output.shape()[3], 6);
}

// -------------------------------------------------------- //
//  Backward pass testleri                                  //
// -------------------------------------------------------- //

TEST(Conv2DBackwardTest, WeightGradShape) {
    nn::Conv2D conv(1, 1, 3, 3);

    tensor input({1, 1, 5, 5}, host, true);
    input.uniform(0.0f, 1.0f);

    tensor output = conv.forward(input);
    tensor loss   = output.sum();
    loss.backward();

    auto grads = conv.getGradients();

    // weight grad shape: (1, 1, 3, 3)
    EXPECT_EQ(grads[0].get().shape()[0], 1);
    EXPECT_EQ(grads[0].get().shape()[1], 1);
    EXPECT_EQ(grads[0].get().shape()[2], 3);
    EXPECT_EQ(grads[0].get().shape()[3], 3);

    // bias grad shape: (1)
    EXPECT_EQ(grads[1].get().len(), 1);
}

TEST(Conv2DBackwardTest, BiasGradValue) {
    // sum(output) backward → ∂L/∂bias = N*oH*oW
    nn::Conv2D conv(1, 1, 3, 3);

    auto params = conv.getParameters();
    params[0].get().fill(1.0f);
    params[1].get().zero();

    tensor input({1, 1, 5, 5}, host, true);
    input.fill(1.0f);

    tensor output = conv.forward(input);  // shape (1,1,3,3)
    tensor loss   = output.sum();
    loss.backward();

    auto grads = conv.getGradients();

    // ∂L/∂bias = oH*oW = 3*3 = 9 (N=1 için)
    EXPECT_NEAR(grads[1].get().get()[0], 9.0f, 1e-3f);
}

TEST(Conv2DBackwardTest, InputGradNotZero) {
    nn::Conv2D conv(1, 1, 3, 3);

    tensor input({1, 1, 5, 5}, host, true);
    input.uniform(0.0f, 1.0f);

    tensor output = conv.forward(input);
    output.sum().backward();

    // Input gradyanı sıfır olmamalı
    EXPECT_GT(input.grad().norm1(), 0.0f);
}