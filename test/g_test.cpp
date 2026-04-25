//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h> // from _deps/googletest-src

TEST(Tensor, DeviceTest) {
    cortex::tensor x({2, 2}, cortex::host);
    ASSERT_EQ(x.device(), cortex::host);
    x = x.to(cortex::cuda);
    ASSERT_EQ(x.device(), cortex::cuda);
}