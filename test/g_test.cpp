//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h>

using namespace cortex;

TEST(DataFrameTest, SplitShapeVerification) {
    // 1. Veri setini yükle ve hazırla
    auto train_df = cortex::load(R"(..\test\archive\antenna_dataset.csv)");
    train_df.one_hot("Fault_Type");

    // Test için target sütunlarını set et
    std::vector<std::string> targets = {
        "Fault_Type_0.000000", "Fault_Type_1.000000",
        "Fault_Type_2.000000", "Fault_Type_3.000000",
        "Fault_Type_4.000000", "Fault_Type_5.000000"
    };
    train_df.Set(targets);

    // 2. Split işlemini gerçekleştir
    auto [x, y] = train_df.split();

    // 3. Beklenen değerleri hesapla
    int64_t expected_rows = train_df.row();
    int64_t expected_x_cols = train_df.col() - static_cast<int64_t>(targets.size());
    int64_t expected_y_cols = static_cast<int64_t>(targets.size());

    // 4. Testler (ASSERT_EQ ile boyutları doğrula)

    // X tensörü için test
    ASSERT_EQ(x.ndim(), 2) << "X tensor should be 2D";
    EXPECT_EQ(x.shape()[0], expected_rows) << "X rows mismatch";
    EXPECT_EQ(x.shape()[1], expected_x_cols) << "X columns mismatch";
    EXPECT_GT(x.len(), 0) << "X tensor should not be empty";

    // Y tensörü için test
    ASSERT_EQ(y.ndim(), 2) << "Y tensor should be 2D";
    EXPECT_EQ(y.shape()[0], expected_rows) << "Y rows mismatch";
    EXPECT_EQ(y.shape()[1], expected_y_cols) << "Y columns mismatch";
    EXPECT_GT(y.len(), 0) << "Y tensor should not be empty";
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