//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h>

using namespace cortex;

TEST(SoftmaxTest, Forward) {
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    tensor input({4}, data);
    input = input.unsqueeze(0);  // [1, 4]

    nn::Softmax softmax;
    tensor output = softmax.forward(input);

    EXPECT_NEAR(output.at(0, 0), 0.0321f, 1e-3f);
    EXPECT_NEAR(output.at(0, 1), 0.0871f, 1e-3f);
    EXPECT_NEAR(output.at(0, 2), 0.2369f, 1e-3f);
    EXPECT_NEAR(output.at(0, 3), 0.6439f, 1e-3f);

    EXPECT_NEAR(output.sum_all(), 1.0f, 1e-5f);
}

TEST(SoftmaxTest, BackwardGradSumZero) {
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    tensor input({1, 4}, data, host, true);

    nn::Softmax softmax;
    tensor output = softmax.forward(input);

    float grad_data[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    tensor upstream_grad({1, 4}, grad_data);

    output.backward(upstream_grad);

    EXPECT_NEAR(input.grad().sum_all(), 0.0f, 1e-5f);

    EXPECT_GT(input.grad().at(0, 0), 0.0f);
    EXPECT_LT(input.grad().at(0, 1), 0.0f);
    EXPECT_LT(input.grad().at(0, 2), 0.0f);
    EXPECT_LT(input.grad().at(0, 3), 0.0f);
}
/*
TEST(SoftmaxTest, BackwardMultiBatch) {
    // 2 satır, 3 sütunlu bir matris [2, 3]
    float data[6] = {1.0f, 2.0f, 3.0f,   10.0f, 20.0f, 30.0f};
    tensor input({2, 3}, data, host, true);

    nn::Softmax softmax;
    tensor output = softmax.forward(input);

    // Her iki satır için de farklı gradyanlar gönderiyoruz
    float grad_data[6] = {1.0f, 0.0f, 0.0f,   0.0f, 1.0f, 0.0f};
    tensor upstream_grad({2, 3}, grad_data);

    output.backward(upstream_grad);

    // EĞER GLOBAL SUM HATASI VARSA:
    // 2. satırdaki devasa sayılar (10, 20, 30) 1. satırın gradyanını tamamen patlatacaktır.
    // EĞER SATIR BAZLI (dim=1) DOĞRU ÇALIŞIYORSA:
    // 1. satırın kendi içindeki gradyan toplamı (örneğin input.grad().at(0, 0) vb.)
    // 2. satırdan tamamen bağımsız ve kararlı kalacaktır.

    // Her satırın kendi içindeki türevsel yönelimlerini ayrı ayrı EXPECT_NEAR ile doğrula.
}
*/

TEST(SoftmaxTest, BackwardMultiBatch) {
    float data[6] = {
        1.0f, 2.0f, 3.0f,
        10.0f, 20.0f, 30.0f
    };

    tensor input({2, 3}, data, host, true);

    nn::Softmax softmax;
    tensor output = softmax.forward(input);

    float grad_data[6] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };

    tensor upstream_grad({2, 3}, grad_data);

    output.backward(upstream_grad);

    float y00 = output.at(0, 0);
    float y01 = output.at(0, 1);
    float y02 = output.at(0, 2);

    float y10 = output.at(1, 0);
    float y11 = output.at(1, 1);
    float y12 = output.at(1, 2);

    EXPECT_NEAR(input.grad().at(0, 0), y00 * (1.0f - y00), 1e-5f);
    EXPECT_NEAR(input.grad().at(0, 1), -y00 * y01,        1e-5f);
    EXPECT_NEAR(input.grad().at(0, 2), -y00 * y02,        1e-5f);

    EXPECT_NEAR(input.grad().at(1, 0), -y10 * y11,        1e-5f);
    EXPECT_NEAR(input.grad().at(1, 1), y11 * (1.0f - y11), 1e-5f);
    EXPECT_NEAR(input.grad().at(1, 2), -y11 * y12,        1e-5f);
}

TEST(ModelGraphTest, SoftmaxWithLossFlow) {
    float input_data[4] = {0.5f, 1.5f, 0.1f, 0.9f};
    tensor input({1, 4}, input_data, host, true);

    nn::Softmax softmax;
    tensor probs = softmax.forward(input);

    loss::CategoricalCrossEntropy cce;
    float target_data[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    tensor target({1, 4}, target_data);

    tensor loss = cce.forward(probs, target);

    loss.backward();
    EXPECT_TRUE(input.has_grad());
}

TEST(DenseLayerTest, WeightGradCheck) {
    float input_data[2] = {1.0f, 2.0f};
    tensor input({1, 2}, input_data, host, true);

    nn::Dense dense(2, 3);


    tensor output = dense.forward(input);

    float grad_data[3] = {1.0f, 1.0f, 1.0f};
    tensor upstream_grad({1, 3}, grad_data);

    output.backward(upstream_grad);

    EXPECT_TRUE(input.has_grad());

    const auto& weights = dense.getParameters()[0].get();

    EXPECT_TRUE(weights.has_grad());

    float grad_sum = 0.0f;
    for(size_t i = 0; i < weights.len(); ++i) {
        grad_sum += std::abs(weights.grad().get()[i]);
    }

    EXPECT_GT(grad_sum, 0.0f);
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