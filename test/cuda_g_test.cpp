//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

using namespace cortex;

class CudaReduceTest : public ::testing::Test {
protected:
    static constexpr size_t N = 8;
    tensor t;

    void SetUp() override {
        std::vector<float32> data(N);
        std::iota(data.begin(), data.end(), 1.0f);

        t = tensor({static_cast<int64>(N)}, data.data(), host);
        t = t.to(cuda);
    }
};

TEST(CudaBasic, DeviceAlloc) {
    const tensor t({8}, cuda);
    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.device(), cuda);
}

TEST(CudaBasic, HostToCuda) {
    const std::vector<float32> data = {1.0f, 2.0f, 3.0f, 4.0f};
    tensor t({4}, data.data(), host);
    t = t.to(cuda);
    EXPECT_EQ(t.device(), cuda);
}

TEST(CudaBasic, CudaToHost) {
    const std::vector<float32> data = {1.0f, 2.0f, 3.0f, 4.0f};
    tensor t({4}, data.data(), host);
    t = t.to(cuda);
    t = t.to(host);
    EXPECT_EQ(t.device(), host);

    for (int64 i = 0; i < 4; ++i) {
        EXPECT_NEAR(t.at(i), data[static_cast<size_t>(i)], 1e-4f);
    }
}

TEST_F(CudaReduceTest, Sum) {
    EXPECT_NEAR(t.sum_all(), 36.0f, 1e-3f);
}

TEST_F(CudaReduceTest, Mean) {
    EXPECT_NEAR(t.mean(), 4.5f, 1e-3f);
}

TEST_F(CudaReduceTest, Min) {
    EXPECT_NEAR(t.min(), 1.0f, 1e-3f);
}

TEST_F(CudaReduceTest, Max) {
    EXPECT_NEAR(t.max(), 8.0f, 1e-3f);
}

TEST_F(CudaReduceTest, Norm2) {
    EXPECT_NEAR(t.norm2(), std::sqrt(204.0f), 1e-2f);
}