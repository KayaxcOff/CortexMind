//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h>

using namespace cortex;

class CudaScalarTest : public ::testing::Test {
protected:
    tensor t;

    void SetUp() override {
        const std::vector data = {
            1.0f, 2.0f, 3.0f, 4.0f
        };

        t = tensor({4}, data.data(), host);
        t = t.to(cuda);
    }

    static void ExpectTensorEq(tensor x, const std::vector<float32>& expected, const float eps = 1e-4f) {
        x = x.to(host);

        ASSERT_EQ(x.len(), expected.size());

        for (int64 i = 0; i < static_cast<int64>(expected.size()); ++i) {
            EXPECT_NEAR(x.at(i), expected[i], eps);
        }
    }
};

TEST_F(CudaScalarTest, ScalarAdd) {
    const tensor r = t + 2.0f;

    EXPECT_EQ(r.device(), cuda);

    ExpectTensorEq(r, {
        3.0f, 4.0f, 5.0f, 6.0f
    });
}

TEST_F(CudaScalarTest, ScalarSub) {
    const tensor r = t - 1.0f;

    ExpectTensorEq(r, {
        0.0f, 1.0f, 2.0f, 3.0f
    });
}

TEST_F(CudaScalarTest, ScalarMul) {
    const tensor r = t * 3.0f;

    ExpectTensorEq(r, {
        3.0f, 6.0f, 9.0f, 12.0f
    });
}

TEST_F(CudaScalarTest, ScalarDiv) {
    const tensor r = t / 2.0f;

    ExpectTensorEq(r, {
        0.5f, 1.0f, 1.5f, 2.0f
    });
}

TEST_F(CudaScalarTest, ReverseScalarSub) {
    const tensor r = 10.0f - t;

    ExpectTensorEq(r, {
        9.0f, 8.0f, 7.0f, 6.0f
    });
}

TEST_F(CudaScalarTest, ReverseScalarMul) {
    const tensor r = 2.0f * t;

    ExpectTensorEq(r, {
        2.0f, 4.0f, 6.0f, 8.0f
    });
}

TEST_F(CudaScalarTest, ScalarAddInplace) {
    t += 5.0f;

    ExpectTensorEq(t, {
        6.0f, 7.0f, 8.0f, 9.0f
    });
}

TEST_F(CudaScalarTest, ScalarMulInplace) {
    t *= 2.0f;

    ExpectTensorEq(t, {
        2.0f, 4.0f, 6.0f, 8.0f
    });
}

TEST_F(CudaScalarTest, ChainedScalarOps) {
    const tensor r = ((t + 2.0f) * 3.0f) / 2.0f;

    ExpectTensorEq(r, {
        4.5f, 6.0f, 7.5f, 9.0f
    });
}

TEST_F(CudaScalarTest, ScalarOpNonContiguous) {
    tensor x({2,2}, cuda);
    x.ones();

    const tensor y = x.transpose();

    tensor r = y + 3.0f;

    r = r.to(host);

    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 2; ++j) {
            EXPECT_NEAR(r.at(i,j), 4.0f, 1e-4f);
        }
    }
}

TEST(CudaScalarStress, LargeTensorAdd) {
    constexpr int64 N = 1 << 20;

    tensor t({N}, cuda);
    t.ones();

    const tensor r = t + 2.0f;

    EXPECT_NEAR(r.mean(), 3.0f, 1e-3f);
}