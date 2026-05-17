//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h>
#include <numeric>

using namespace cortex;

class ScalarTest : public ::testing::Test {
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
//  Out-of-place                                            //
// -------------------------------------------------------- //

TEST_F(ScalarTest, AddOutOfPlace) {
    const tensor result = t + 10.0f;
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    static_cast<float32>(i + 1) + 10.0f, 1e-4f);
    }
}

TEST_F(ScalarTest, SubOutOfPlace) {
    const tensor result = t - 1.0f;
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    static_cast<float32>(i + 1) - 1.0f, 1e-4f);
    }
}

TEST_F(ScalarTest, MulOutOfPlace) {
    const tensor result = t * 2.0f;
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    static_cast<float32>(i + 1) * 2.0f, 1e-4f);
    }
}

TEST_F(ScalarTest, DivOutOfPlace) {
    const tensor result = t / 2.0f;
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    static_cast<float32>(i + 1) / 2.0f, 1e-4f);
    }
}

// -------------------------------------------------------- //
//  In-place                                                //
// -------------------------------------------------------- //

TEST_F(ScalarTest, AddInPlace) {
    t += 10.0f;
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(t.at(static_cast<int64>(i)),
                    static_cast<float32>(i + 1) + 10.0f, 1e-4f);
    }
}

TEST_F(ScalarTest, SubInPlace) {
    t -= 1.0f;
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(t.at(static_cast<int64>(i)),
                    static_cast<float32>(i + 1) - 1.0f, 1e-4f);
    }
}

TEST_F(ScalarTest, MulInPlace) {
    t *= 3.0f;
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(t.at(static_cast<int64>(i)),
                    static_cast<float32>(i + 1) * 3.0f, 1e-4f);
    }
}

TEST_F(ScalarTest, DivInPlace) {
    t /= 4.0f;
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(t.at(static_cast<int64>(i)),
                    static_cast<float32>(i + 1) / 4.0f, 1e-4f);
    }
}

// -------------------------------------------------------- //
//  Edge cases                                              //
// -------------------------------------------------------- //

TEST_F(ScalarTest, AddZero) {
    const tensor result = t + 0.0f;
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    static_cast<float32>(i + 1), 1e-4f);
    }
}

TEST_F(ScalarTest, MulOne) {
    const tensor result = t * 1.0f;
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    static_cast<float32>(i + 1), 1e-4f);
    }
}

TEST_F(ScalarTest, MulZero) {
    const tensor result = t * 0.0f;
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)), 0.0f, 1e-4f);
    }
}

TEST_F(ScalarTest, SubFromLeft) {
    // value - tensor
    const tensor result = 10.0f - t;
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    10.0f - static_cast<float32>(i + 1), 1e-4f);
    }
}

TEST_F(ScalarTest, MulFromLeft) {
    // value * tensor
    const tensor result = 3.0f * t;
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(result.at(static_cast<int64>(i)),
                    3.0f * static_cast<float32>(i + 1), 1e-4f);
    }
}

TEST_F(ScalarTest, NonMultipleOf8) {
    // remainder path — N=5
    const std::vector<float32> data = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
    const tensor t5({5}, data.data(), host);
    const tensor result = t5 * 0.5f;
    for (int64 i = 0; i < 5; ++i) {
        EXPECT_NEAR(result.at(i), data[static_cast<size_t>(i)] * 0.5f, 1e-4f);
    }
}