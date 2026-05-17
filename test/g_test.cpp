//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h>
#include <numeric>

using namespace cortex;

class ElementWiseTest : public ::testing::Test {
protected:
    static constexpr size_t N = 8;
    tensor t;

    void SetUp() override {
        // 1, 2, 3, 4, 5, 6, 7, 8
        std::vector<float32> data(N);
        std::iota(data.begin(), data.end(), 1.0f);
        t = tensor({static_cast<int64>(N)}, data.data(), host);
    }
};

// -------------------------------------------------------- //
//  Unary ops                                               //
// -------------------------------------------------------- //

TEST_F(ElementWiseTest, Log) {
    const tensor result = t.log();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    std::log(static_cast<float32>(i + 1)), 1e-4f);
    }
}

TEST_F(ElementWiseTest, Exp) {
    // küçük değerler kullan — overflow riski yok
    const std::vector<float32> data = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f};
    const tensor te({8}, data.data(), host);
    const tensor result = te.exp();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    std::exp(data[i]), 1e-3f);
    }
}

TEST_F(ElementWiseTest, Sqrt) {
    const tensor result = t.sqrt();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    std::sqrt(static_cast<float32>(i + 1)), 1e-4f);
    }
}

TEST_F(ElementWiseTest, Rsqrt) {
    const tensor result = t.rsqrt();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    1.0f / std::sqrt(static_cast<float32>(i + 1)), 1e-3f);
    }
}

TEST_F(ElementWiseTest, Pow) {
    const tensor result = t.pow(2.0f);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    std::pow(static_cast<float32>(i + 1), 2.0f), 1e-3f);
    }
}

TEST_F(ElementWiseTest, Abs) {
    const std::vector<float32> data = {-4.0f, -3.0f, -2.0f, -1.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const tensor ta({8}, data.data(), host);
    const tensor result = ta.abs();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    std::abs(data[i]), 1e-4f);
    }
}

TEST_F(ElementWiseTest, Neg) {
    const tensor result = t.neg();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    -static_cast<float32>(i + 1), 1e-4f);
    }
}

TEST_F(ElementWiseTest, Sin) {
    const tensor result = t.sin();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    std::sin(static_cast<float32>(i + 1)), 1e-4f);
    }
}

TEST_F(ElementWiseTest, Cos) {
    const tensor result = t.cos();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    std::cos(static_cast<float32>(i + 1)), 1e-4f);
    }
}

TEST_F(ElementWiseTest, Sign) {
    const std::vector<float32> data = {-3.0f, -1.0f, 0.0f, 2.0f, 4.0f, -5.0f, 6.0f, -7.0f};
    const tensor ts({8}, data.data(), host);
    const tensor result = ts.sign();
    const std::vector<float32> expected = {-1.0f, -1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f};
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)), expected[i], 1e-4f);
    }
}

// -------------------------------------------------------- //
//  Chaining                                                //
// -------------------------------------------------------- //

TEST_F(ElementWiseTest, SqrtOfSquare) {
    // sqrt(x^2) = x için x > 0
    const tensor result = t.pow(2.0f).sqrt();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    static_cast<float32>(i + 1), 1e-3f);
    }
}

TEST_F(ElementWiseTest, ExpOfLog) {
    // exp(log(x)) = x
    const tensor result = t.log().exp();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    static_cast<float32>(i + 1), 1e-3f);
    }
}

// -------------------------------------------------------- //
//  Remainder path — N=5                                    //
// -------------------------------------------------------- //

TEST_F(ElementWiseTest, SqrtNonMultipleOf8) {
    const std::vector<float32> data = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f};
    const tensor t5({5}, data.data(), host);
    const tensor result = t5.sqrt();
    const std::vector<float32> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    for (int64 i = 0; i < 5; ++i) {
        EXPECT_NEAR(result.at(i), expected[static_cast<size_t>(i)], 1e-4f);
    }
}