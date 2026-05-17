//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

using namespace cortex;

class ReduceTest : public ::testing::Test {
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

TEST(ReduceSimple, AllocOnly) {
    const tensor t({8}, host);
    EXPECT_FALSE(t.empty());
}

TEST(ReduceSimple, DataConstructor) {
    const std::vector data = {1.0f, 2.0f, 3.0f};
    const tensor t({3}, data.data(), host);
    EXPECT_FALSE(t.empty());
}

TEST(ReduceSimple, DataAccess) {
    const std::vector data = {1.0f, 2.0f, 3.0f};
    tensor t({3}, data.data(), host);
    EXPECT_NEAR(t.at(0), 1.0f, 1e-4f);
}

TEST(ReduceSimple, SumSmall) {
    const std::vector data = {1.0f, 2.0f, 3.0f};
    const tensor t({3}, data.data(), host);
    EXPECT_NEAR(t.sum_all(), 6.0f, 1e-4f);
}

TEST_F(ReduceTest, Sum) {
    // 1+2+...+8 = 36
    EXPECT_NEAR(t.sum_all(), 36.0f, 1e-4f);
}

TEST_F(ReduceTest, Mean) {
    // 36 / 8 = 4.5
    EXPECT_NEAR(t.mean(), 4.5f, 1e-4f);
}

TEST_F(ReduceTest, Variance) {
    // population variance: E[(x - mean)^2]
    // mean = 4.5
    // diffs: -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5
    // squared: 12.25, 6.25, 2.25, 0.25, 0.25, 2.25, 6.25, 12.25
    // sum = 42, / 8 = 5.25
    EXPECT_NEAR(t.variance(), 5.25f, 1e-4f);
}

TEST_F(ReduceTest, Stdv) {
    EXPECT_NEAR(t.stdv(), std::sqrt(5.25f), 1e-4f);
}

TEST_F(ReduceTest, Min) {
    EXPECT_NEAR(t.min(), 1.0f, 1e-4f);
}

TEST_F(ReduceTest, Max) {
    EXPECT_NEAR(t.max(), 8.0f, 1e-4f);
}

TEST_F(ReduceTest, Norm1) {
    // |1|+|2|+...+|8| = 36
    EXPECT_NEAR(t.norm1(), 36.0f, 1e-4f);
}

TEST_F(ReduceTest, Norm2) {
    // sqrt(1^2 + 2^2 + ... + 8^2) = sqrt(204)
    EXPECT_NEAR(t.norm2(), std::sqrt(204.0f), 1e-4f);
}

TEST_F(ReduceTest, SumNonMultipleOf8) {
    // 8'in katı olmayan N — remainder path test
    const std::vector data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    const tensor t5({5}, data.data(), host);
    EXPECT_NEAR(t5.sum_all(), 15.0f, 1e-4f);
}

TEST_F(ReduceTest, MeanSingleElement) {
    tensor t1({1}, nullptr, host);
    t1.fill(7.0f);
    EXPECT_NEAR(t1.mean(), 7.0f, 1e-4f);
}

TEST_F(ReduceTest, MinNegative) {
    const std::vector data = {-3.0f, -1.0f, 0.0f, 2.0f};
    const tensor tn({4}, data.data(), host);
    EXPECT_NEAR(tn.min(), -3.0f, 1e-4f);
}

TEST_F(ReduceTest, MaxNegative) {
    const std::vector data = {-3.0f, -1.0f, 0.0f, 2.0f};
    const tensor tn({4}, data.data(), host);
    EXPECT_NEAR(tn.max(), 2.0f, 1e-4f);
}