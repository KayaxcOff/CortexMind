//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h>
#include <numeric>

using namespace cortex;

// ========================================================== //
// Helper Class for CUDA Binary Operations Tests
// ========================================================== //

class CudaBinaryTest : public ::testing::Test {
protected:
    tensor a_host, b_host;

    void SetUp() override {
        // a = [[1, 2, 3, 4],
        //      [5, 6, 7, 8]]  shape(2,4)
        const std::vector<float32> data_a = {1, 2, 3, 4, 5, 6, 7, 8};
        a_host = tensor({2, 4}, data_a.data(), host);

        // b = [[1, 1, 1, 1],
        //      [2, 2, 2, 2]]  shape(2,4)
        const std::vector<float32> data_b = {1, 1, 1, 1, 2, 2, 2, 2};
        b_host = tensor({2, 4}, data_b.data(), host);
    }

    static void ExpectMatrix2DNear(const tensor& result,
                                    const std::vector<float32>& expected,
                                    const float eps = 1e-4f) {
        tensor result_host = result.to(host);

        ASSERT_EQ(result_host.len(), expected.size());

        for (size_t idx = 0; idx < expected.size(); ++idx) {
            EXPECT_NEAR(result_host.get()[idx], expected[idx], eps)
                << "Mismatch at linear index " << idx;
        }
    }
};

// ========================================================== //
// Element-wise Binary Operations - Same Shape
// ========================================================== //

TEST_F(CudaBinaryTest, Add) {
    const tensor a_cuda = a_host.to(cuda);
    const tensor b_cuda = b_host.to(cuda);
    const tensor result = a_cuda + b_cuda;

    // [[2,3,4,5],[7,8,9,10]]
    const std::vector<float32> expected = {2, 3, 4, 5, 7, 8, 9, 10};
    ExpectMatrix2DNear(result, expected);
}

TEST_F(CudaBinaryTest, Sub) {
    const tensor a_cuda = a_host.to(cuda);
    const tensor b_cuda = b_host.to(cuda);
    const tensor result = a_cuda - b_cuda;

    // [[0,1,2,3],[3,4,5,6]]
    const std::vector<float32> expected = {0, 1, 2, 3, 3, 4, 5, 6};
    ExpectMatrix2DNear(result, expected);
}

TEST_F(CudaBinaryTest, Mul) {
    const tensor a_cuda = a_host.to(cuda);
    const tensor b_cuda = b_host.to(cuda);
    const tensor result = a_cuda * b_cuda;

    // [[1,2,3,4],[10,12,14,16]]
    const std::vector<float32> expected = {1, 2, 3, 4, 10, 12, 14, 16};
    ExpectMatrix2DNear(result, expected);
}

TEST_F(CudaBinaryTest, Div) {
    const tensor a_cuda = a_host.to(cuda);
    const tensor b_cuda = b_host.to(cuda);
    const tensor result = a_cuda / b_cuda;

    // [[1,2,3,4],[2.5,3,3.5,4]]
    const std::vector<float32> expected = {1, 2, 3, 4, 2.5f, 3, 3.5f, 4};
    ExpectMatrix2DNear(result, expected);
}

// ========================================================== //
// In-Place Operations
// ========================================================== //

TEST_F(CudaBinaryTest, AddInPlace) {
    tensor a_cuda = a_host.clone().to(cuda);
    const tensor b_cuda = b_host.to(cuda);
    a_cuda += b_cuda;

    const std::vector<float32> expected = {2, 3, 4, 5, 7, 8, 9, 10};
    ExpectMatrix2DNear(a_cuda, expected);
}

TEST_F(CudaBinaryTest, SubInPlace) {
    tensor a_cuda = a_host.clone().to(cuda);
    const tensor b_cuda = b_host.to(cuda);
    a_cuda -= b_cuda;

    const std::vector<float32> expected = {0, 1, 2, 3, 3, 4, 5, 6};
    ExpectMatrix2DNear(a_cuda, expected);
}

TEST_F(CudaBinaryTest, MulInPlace) {
    tensor a_cuda = a_host.clone().to(cuda);
    const tensor b_cuda = b_host.to(cuda);
    a_cuda *= b_cuda;

    const std::vector<float32> expected = {1, 2, 3, 4, 10, 12, 14, 16};
    ExpectMatrix2DNear(a_cuda, expected);
}

TEST_F(CudaBinaryTest, DivInPlace) {
    tensor a_cuda = a_host.clone().to(cuda);
    const tensor b_cuda = b_host.to(cuda);
    a_cuda /= b_cuda;

    const std::vector<float32> expected = {1, 2, 3, 4, 2.5f, 3, 3.5f, 4};
    ExpectMatrix2DNear(a_cuda, expected);
}

// ========================================================== //
// Broadcasting - Row Broadcast
// ========================================================== //

TEST_F(CudaBinaryTest, RowBroadcastAdd) {
    const std::vector<float32> data_y = {10, 20, 30, 40};
    tensor y_host({4}, data_y.data(), host);

    tensor a_cuda = a_host.to(cuda);
    tensor y_cuda = y_host.to(cuda);
    tensor result = a_cuda + y_cuda;

    // [[11,22,33,44],[15,26,37,48]]
    const std::vector<float32> expected = {11, 22, 33, 44, 15, 26, 37, 48};
    ExpectMatrix2DNear(result, expected);
}

TEST_F(CudaBinaryTest, RowBroadcastMul) {
    const std::vector<float32> data_y = {2, 2, 2, 2};
    tensor y_host({4}, data_y.data(), host);

    tensor a_cuda = a_host.to(cuda);
    tensor y_cuda = y_host.to(cuda);
    tensor result = a_cuda * y_cuda;

    // [[2,4,6,8],[10,12,14,16]]
    const std::vector<float32> expected = {2, 4, 6, 8, 10, 12, 14, 16};
    ExpectMatrix2DNear(result, expected);
}

// ========================================================== //
// Broadcasting - Column Broadcast
// ========================================================== //

TEST_F(CudaBinaryTest, ColBroadcastAdd) {
    const std::vector<float32> data_y = {10, 20};
    tensor y_host({2, 1}, data_y.data(), host);

    tensor a_cuda = a_host.to(cuda);
    tensor y_cuda = y_host.to(cuda);
    tensor result = a_cuda + y_cuda;

    // [[11,12,13,14],[25,26,27,28]]
    const std::vector<float32> expected = {11, 12, 13, 14, 25, 26, 27, 28};
    ExpectMatrix2DNear(result, expected);
}

TEST_F(CudaBinaryTest, ColBroadcastMul) {
    const std::vector<float32> data_y = {2, 3};
    tensor y_host({2, 1}, data_y.data(), host);

    tensor a_cuda = a_host.to(cuda);
    tensor y_cuda = y_host.to(cuda);
    tensor result = a_cuda * y_cuda;

    // [[2,4,6,8],[15,18,21,24]]
    const std::vector<float32> expected = {2, 4, 6, 8, 15, 18, 21, 24};
    ExpectMatrix2DNear(result, expected);
}

// ========================================================== //
// Matrix Multiplication
// ========================================================== //

TEST(CudaMatmulTest, BasicMatmul) {
    // (2,3) @ (3,2) = (2,2)
    const std::vector<float32> data_a = {1, 2, 3, 4, 5, 6};
    const std::vector<float32> data_b = {7, 8, 9, 10, 11, 12};

    tensor a_host({2, 3}, data_a.data(), host);
    tensor b_host({3, 2}, data_b.data(), host);

    tensor a_cuda = a_host.to(cuda);
    tensor b_cuda = b_host.to(cuda);
    tensor result = a_cuda.matmul(b_cuda);

    // [[1*7+2*9+3*11, 1*8+2*10+3*12],
    //  [4*7+5*9+6*11, 4*8+5*10+6*12]]
    // = [[58, 64], [139, 154]]
    tensor result_host = result.to(host);
    EXPECT_NEAR(result_host.at(0, 0), 58.0f, 1e-3f);
    EXPECT_NEAR(result_host.at(0, 1), 64.0f, 1e-3f);
    EXPECT_NEAR(result_host.at(1, 0), 139.0f, 1e-3f);
    EXPECT_NEAR(result_host.at(1, 1), 154.0f, 1e-3f);
}

TEST(CudaMatmulTest, MatmulIdentity) {
    // A @ I = A
    const std::vector<float32> data_a = {1, 2, 3, 4};
    const std::vector<float32> data_i = {1, 0, 0, 1};

    tensor a_host({2, 2}, data_a.data(), host);
    tensor eye_host({2, 2}, data_i.data(), host);

    tensor a_cuda = a_host.to(cuda);
    tensor eye_cuda = eye_host.to(cuda);
    tensor result = a_cuda.matmul(eye_cuda);

    tensor result_host = result.to(host);
    for (int64 i = 0; i < 2; ++i) {
        for (int64 j = 0; j < 2; ++j) {
            EXPECT_NEAR(result_host.at(i, j), data_a[i * 2 + j], 1e-3f);
        }
    }
}

TEST(CudaMatmulTest, MatmulSquare) {
    // (4,4) @ (4,4)
    std::vector<float32> data(16);
    std::iota(data.begin(), data.end(), 1.0f);

    tensor a_host({4, 4}, data.data(), host);
    tensor a_cuda = a_host.to(cuda);
    tensor result = a_cuda.matmul(a_cuda);

    tensor result_host = result.to(host);

    // a[0][0] = 1*1+2*5+3*9+4*13 = 1+10+27+52 = 90
    EXPECT_NEAR(result_host.at(0, 0), 90.0f, 1e-2f);
}

TEST(CudaMatmulTest, MatmulRectangular) {
    // (3,2) @ (2,3) = (3,3)
    const std::vector<float32> data_a = {1, 2, 3, 4, 5, 6};
    const std::vector<float32> data_b = {7, 8, 9, 10, 11, 12};

    tensor a_host({3, 2}, data_a.data(), host);
    tensor b_host({2, 3}, data_b.data(), host);

    tensor a_cuda = a_host.to(cuda);
    tensor b_cuda = b_host.to(cuda);
    tensor result = a_cuda.matmul(b_cuda);

    tensor result_host = result.to(host);

    // a[0][0] = 1*7 + 2*10 = 7 + 20 = 27
    EXPECT_NEAR(result_host.at(0, 0), 27.0f, 1e-3f);
    // a[0][1] = 1*8 + 2*11 = 8 + 22 = 30
    EXPECT_NEAR(result_host.at(0, 1), 30.0f, 1e-3f);
}

// ========================================================== //
// Mixed Device Operations
// ========================================================== //

TEST(CudaMixedDeviceTest, CudaToHostTransfer) {
    const std::vector<float32> data = {1, 2, 3, 4};
    const tensor t_host({2, 2}, data.data(), host);

    const tensor t_cuda = t_host.to(cuda);
    tensor t_back = t_cuda.to(host);

    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(t_back.get()[i], data[i], 1e-6f);
    }
}

TEST(CudaMixedDeviceTest, ChainedOperations) {
    const std::vector<float32> data_a = {1, 2, 3, 4};
    const std::vector<float32> data_b = {5, 6, 7, 8};

    tensor a_host({2, 2}, data_a.data(), host);
    tensor b_host({2, 2}, data_b.data(), host);

    tensor result = a_host.to(cuda) + b_host.to(cuda);

    tensor result_host = result.to(host);

    const std::vector<float32> expected = {6, 8, 10, 12};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_host.get()[i], expected[i], 1e-4f);
    }
}

// ========================================================== //
// Stress Tests - Large Tensors
// ========================================================== //

TEST(CudaStressTest, LargeTensorAdd) {
    constexpr size_t N = 1024 * 256;  // 256K elements

    std::vector data_a(N, 1.0f);
    std::vector data_b(N, 2.0f);

    tensor a_host({N}, data_a.data(), host);
    tensor b_host({N}, data_b.data(), host);

    tensor a_cuda = a_host.to(cuda);
    tensor b_cuda = b_host.to(cuda);
    tensor result = a_cuda + b_cuda;

    tensor result_host = result.to(host);

    for (size_t i = 0; i < 100; i += 10) {
        EXPECT_NEAR(result_host.get()[i], 3.0f, 1e-4f);
    }
}

TEST(CudaStressTest, LargeTensor2D) {
    constexpr size_t rows = 512;
    constexpr size_t cols = 512;

    std::vector data_a(rows * cols, 1.5f);
    std::vector data_b(rows * cols, 2.5f);

    tensor a_host({static_cast<int64>(rows), static_cast<int64>(cols)}, data_a.data(), host);
    tensor b_host({static_cast<int64>(rows), static_cast<int64>(cols)}, data_b.data(), host);

    tensor a_cuda = a_host.to(cuda);
    tensor b_cuda = b_host.to(cuda);
    tensor result = a_cuda * b_cuda;

    tensor result_host = result.to(host);

    EXPECT_NEAR(result_host.get()[0], 3.75f, 1e-3f);
}

// ========================================================== //
// Advanced Broadcasting Tests
// ========================================================== //

TEST(CudaBroadcastAdvancedTest, Broadcast1DTo2D) {
    const std::vector data_a = {5.0f};
    const std::vector<float32> data_b = {1, 2, 3, 4, 5, 6};

    tensor a_host({1}, data_a.data(), host);
    tensor b_host({2, 3}, data_b.data(), host);

    tensor a_cuda = a_host.to(cuda);
    tensor b_cuda = b_host.to(cuda);
    tensor result = a_cuda + b_cuda;

    tensor result_host = result.to(host);
    const std::vector<float32> expected = {6, 7, 8, 9, 10, 11};

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_host.get()[i], expected[i], 1e-4f);
    }
}

TEST(CudaBroadcastAdvancedTest, Broadcast1DScalar) {
    const std::vector data_a = {3.0f};
    const std::vector<float32> data_b = {2, 4, 6, 8, 10, 12};

    tensor a_host({1}, data_a.data(), host);
    tensor b_host({2, 3}, data_b.data(), host);

    tensor a_cuda = a_host.to(cuda);
    tensor b_cuda = b_host.to(cuda);
    tensor result = a_cuda * b_cuda;

    tensor result_host = result.to(host);
    const std::vector<float32> expected = {6, 12, 18, 24, 30, 36};

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_host.get()[i], expected[i], 1e-4f);
    }
}