//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h>
#include <numeric>

using namespace cortex;

class MatrixTest : public ::testing::Test {
protected:
    tensor a, b;

    void SetUp() override {
        // a = [[1, 2, 3, 4],
        //      [5, 6, 7, 8]]  shape(2,4)
        const std::vector<float32> data_a = {1,2,3,4,5,6,7,8};
        a = tensor({2, 4}, data_a.data(), host);

        // b = [[1, 1, 1, 1],
        //      [2, 2, 2, 2]]  shape(2,4)
        const std::vector<float32> data_b = {1,1,1,1,2,2,2,2};
        b = tensor({2, 4}, data_b.data(), host);
    }
};

// -------------------------------------------------------- //
//  Element-wise binary ops — same shape                   //
// -------------------------------------------------------- //

TEST_F(MatrixTest, Add) {
    const tensor result = a + b;
    // [[2,3,4,5],[7,8,9,10]]
    const std::vector<float32> expected = {2,3,4,5,7,8,9,10};
    for (int64 i = 0; i < 2; ++i)
        for (int64 j = 0; j < 4; ++j)
            EXPECT_NEAR(result.at(i,j), expected[i*4+j], 1e-4f);
}

TEST_F(MatrixTest, Sub) {
    const tensor result = a - b;
    // [[0,1,2,3],[3,4,5,6]]
    const std::vector<float32> expected = {0,1,2,3,3,4,5,6};
    for (int64 i = 0; i < 2; ++i)
        for (int64 j = 0; j < 4; ++j)
            EXPECT_NEAR(result.at(i,j), expected[i*4+j], 1e-4f);
}

TEST_F(MatrixTest, Mul) {
    const tensor result = a * b;
    // [[1,2,3,4],[10,12,14,16]]
    const std::vector<float32> expected = {1,2,3,4,10,12,14,16};
    for (int64 i = 0; i < 2; ++i)
        for (int64 j = 0; j < 4; ++j)
            EXPECT_NEAR(result.at(i,j), expected[i*4+j], 1e-4f);
}

TEST_F(MatrixTest, Div) {
    const tensor result = a / b;
    // [[1,2,3,4],[2.5,3,3.5,4]]
    const std::vector<float32> expected = {1,2,3,4,2.5f,3,3.5f,4};
    for (int64 i = 0; i < 2; ++i)
        for (int64 j = 0; j < 4; ++j)
            EXPECT_NEAR(result.at(i,j), expected[i*4+j], 1e-4f);
}

// -------------------------------------------------------- //
//  In-place                                                //
// -------------------------------------------------------- //

TEST_F(MatrixTest, AddInPlace) {
    a += b;
    const std::vector<float32> expected = {2,3,4,5,7,8,9,10};
    for (int64 i = 0; i < 2; ++i)
        for (int64 j = 0; j < 4; ++j)
            EXPECT_NEAR(a.at(i,j), expected[i*4+j], 1e-4f);
}

TEST_F(MatrixTest, MulInPlace) {
    a *= b;
    const std::vector<float32> expected = {1,2,3,4,10,12,14,16};
    for (int64 i = 0; i < 2; ++i)
        for (int64 j = 0; j < 4; ++j)
            EXPECT_NEAR(a.at(i,j), expected[i*4+j], 1e-4f);
}

// -------------------------------------------------------- //
//  Broadcast                                               //
// -------------------------------------------------------- //

TEST_F(MatrixTest, RowBroadcastAdd) {
    // a(2,4) + y(4) — her satıra y eklenir
    const std::vector<float32> data_y = {10, 20, 30, 40};
    const tensor y({4}, data_y.data(), host);
    const tensor result = a + y;
    // [[11,22,33,44],[15,26,37,48]]
    const std::vector<float32> expected = {11,22,33,44,15,26,37,48};
    for (int64 i = 0; i < 2; ++i)
        for (int64 j = 0; j < 4; ++j)
            EXPECT_NEAR(result.at(i,j), expected[i*4+j], 1e-4f);
}

TEST_F(MatrixTest, ColBroadcastAdd) {
    // a(2,4) + y(2,1) — her sütuna y eklenir
    const std::vector<float32> data_y = {10, 20};
    const tensor y({2, 1}, data_y.data(), host);
    const tensor result = a + y;
    // [[11,12,13,14],[25,26,27,28]]
    const std::vector<float32> expected = {11,12,13,14,25,26,27,28};
    for (int64 i = 0; i < 2; ++i)
        for (int64 j = 0; j < 4; ++j)
            EXPECT_NEAR(result.at(i,j), expected[i*4+j], 1e-4f);
}

// -------------------------------------------------------- //
//  Matmul                                                  //
// -------------------------------------------------------- //

TEST(MatmulTest, Basic) {
    // (2,3) @ (3,2) = (2,2)
    const std::vector<float32> data_a = {1,2,3,4,5,6};
    const std::vector<float32> data_b = {7,8,9,10,11,12};
    const tensor a({2,3}, data_a.data(), host);
    const tensor b({3,2}, data_b.data(), host);
    const tensor result = a.matmul(b);

    // [[1*7+2*9+3*11, 1*8+2*10+3*12],
    //  [4*7+5*9+6*11, 4*8+5*10+6*12]]
    // = [[58, 64], [139, 154]]
    EXPECT_NEAR(result.at(0,0), 58.0f,  1e-3f);
    EXPECT_NEAR(result.at(0,1), 64.0f,  1e-3f);
    EXPECT_NEAR(result.at(1,0), 139.0f, 1e-3f);
    EXPECT_NEAR(result.at(1,1), 154.0f, 1e-3f);
}

TEST(MatmulTest, Identity) {
    // A @ I = A
    const std::vector<float32> data_a = {1,2,3,4};
    const std::vector<float32> data_i = {1,0,0,1};
    const tensor a({2,2}, data_a.data(), host);
    const tensor eye({2,2}, data_i.data(), host);
    const tensor result = a.matmul(eye);

    for (int64 i = 0; i < 2; ++i)
        for (int64 j = 0; j < 2; ++j)
            EXPECT_NEAR(result.at(i,j), data_a[i*2+j], 1e-3f);
}

TEST(MatmulTest, Square) {
    // (4,4) @ (4,4)
    std::vector<float32> data(16);
    std::iota(data.begin(), data.end(), 1.0f);
    const tensor a({4,4}, data.data(), host);
    const tensor result = a.matmul(a);

    // sadece köşegen elemanları kontrol et
    // a[0][0] = 1*1+2*5+3*9+4*13 = 1+10+27+52 = 90
    EXPECT_NEAR(result.at(0,0), 90.0f, 1e-2f);
}