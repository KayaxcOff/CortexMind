//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h> // from _deps/googletest-src

TEST(CXM_CUDA_AVAILABLE_TEST, CompileTimeGateWorks) {
    #if !CXM_IS_CUDA_AVAILABLE
        EXPECT_FALSE(cortex::has_cuda());
    #else //#if !CXM_IS_CUDA_AVAILABLE
        SUCCEED();
    #endif //#if !CXM_IS_CUDA_AVAILABLE #else
}

TEST(CudaAvailabilityTest, IsDeterministic) {
    const bool first = cortex::has_cuda();
    const bool second = cortex::has_cuda();

    EXPECT_EQ(first, second);
}

TEST(CudaAvailabilityTest, TrueImpliesCompileFlag) {
    if (cortex::has_cuda()) {
        EXPECT_TRUE(CXM_IS_CUDA_AVAILABLE);
    }
}