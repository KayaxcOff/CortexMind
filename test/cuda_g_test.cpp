//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

using namespace cortex;

float32 pi = std::numbers::pi_v<float32>;
float32 half_pi = std::numbers::pi_v<float32> / 2.0f;

class CudaUnaryTest : public ::testing::Test {
protected:
    tensor t;

    void SetUp() override {
        const std::vector data = {
            1.0f,
            4.0f,
            9.0f,
            16.0f
        };

        t = tensor({4}, data.data(), host).to(cuda);
    }

    static void ExpectTensorNear(tensor x, const std::vector<float32>& expected, const float eps = 1e-4f) {
        x = x.to(host);

        ASSERT_EQ(x.len(), expected.size());

        for (int64 i = 0; i < static_cast<int64>(expected.size()); ++i) {
            EXPECT_NEAR(x.at(i), expected[i], eps);
        }
    }
};

TEST_F(CudaUnaryTest, Sqrt) {
    const tensor r = t.sqrt();

    EXPECT_EQ(r.device(), cuda);

    ExpectTensorNear(r, {
        1.0f,
        2.0f,
        3.0f,
        4.0f
    });
}

TEST_F(CudaUnaryTest, RSqrt) {
    const tensor r = t.rsqrt();

    ExpectTensorNear(r, {
        1.0f,
        0.5f,
        1.0f / 3.0f,
        0.25f
    }, 1e-3f);
}

TEST_F(CudaUnaryTest, Pow) {
    const tensor r = t.pow(2.0f);

    ExpectTensorNear(r, {
        1.0f,
        16.0f,
        81.0f,
        256.0f
    });
}

TEST(CudaUnaryOps, Exp) {
    const std::vector data = {
        0.0f,
        1.0f,
        2.0f
    };

    tensor t({3}, data.data(), host);
    t = t.to(cuda);

    tensor r = t.exp();
    r = r.to(host);

    EXPECT_NEAR(r.at(0), std::exp(0.0f), 1e-4f);
    EXPECT_NEAR(r.at(1), std::exp(1.0f), 1e-4f);
    EXPECT_NEAR(r.at(2), std::exp(2.0f), 1e-4f);
}

TEST(CudaUnaryOps, Log) {
    const std::vector data = {
        1.0f,
        2.7182818f,
        7.389056f
    };

    tensor t({3}, data.data(), host);
    t = t.to(cuda);

    tensor r = t.log();
    r = r.to(host);

    EXPECT_NEAR(r.at(0), 0.0f, 1e-4f);
    EXPECT_NEAR(r.at(1), 1.0f, 1e-3f);
    EXPECT_NEAR(r.at(2), 2.0f, 1e-3f);
}

TEST(CudaUnaryOps, Sin) {
    const std::vector data = {
        0.0f,
        half_pi,
        pi
    };

    tensor t({3}, data.data(), host);
    t = t.to(cuda);

    tensor r = t.sin();
    r = r.to(host);

    EXPECT_NEAR(r.at(0), 0.0f, 1e-4f);
    EXPECT_NEAR(r.at(1), 1.0f, 1e-4f);
    EXPECT_NEAR(r.at(2), 0.0f, 1e-4f);
}

TEST(CudaUnaryOps, Cos) {
    const std::vector data = {
        0.0f,
        pi
    };

    tensor t({2}, data.data(), host);
    t = t.to(cuda);

    tensor r = t.cos();
    r = r.to(host);

    EXPECT_NEAR(r.at(0), 1.0f, 1e-4f);
    EXPECT_NEAR(r.at(1), -1.0f, 1e-4f);
}
/*
TEST(CudaUnaryOps, Abs) {
    const std::vector data = {
        -1.0f,
        -2.0f,
        3.0f
    };

    tensor t({3}, data.data(), host);
    t = t.to(cuda);

    tensor r = t.abs();

    ExpectTensorNear(r, {
        1.0f,
        2.0f,
        3.0f
    });
}

TEST(CudaUnaryOps, Neg) {
    const std::vector data = {
        1.0f,
        -2.0f,
        3.0f
    };

    tensor t({3}, data.data(), host);
    t = t.to(cuda);

    tensor r = t.neg();

    ExpectTensorNear(r, {
        -1.0f,
        2.0f,
        -3.0f
    });
}

TEST(CudaUnaryOps, Sign) {
    const std::vector data = {
        -5.0f,
        0.0f,
        8.0f
    };

    tensor t({3}, data.data(), host);
    t = t.to(cuda);

    tensor r = t.sign();

    ExpectTensorNear(r, {
        -1.0f,
        0.0f,
        1.0f
    });
}
*/
TEST(CudaUnaryAdvanced, UnaryOnTranspose) {
    tensor t({2,2}, cuda);
    t.fill(4.0f);

    const tensor x = t.transpose();

    tensor r = x.sqrt();

    r = r.to(host);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(r.at(i,j), 2.0f, 1e-4f);
        }
    }
}

TEST(CudaUnaryStress, LargeExp) {
    constexpr int64 N = 1 << 20;

    tensor t({N}, cuda);
    t.zero();

    const tensor r = t.exp();

    EXPECT_NEAR(r.mean(), 1.0f, 1e-3f);
}

TEST(CudaUnaryNumerics, SmallValues) {
    const std::vector data = {
        1e-6f,
        1e-4f,
        1e-2f
    };

    tensor t({3}, data.data(), host);
    t = t.to(cuda);

    tensor r = t.rsqrt();
    r = r.to(host);

    EXPECT_TRUE(std::isfinite(r.at(0)));
    EXPECT_TRUE(std::isfinite(r.at(1)));
    EXPECT_TRUE(std::isfinite(r.at(2)));
}