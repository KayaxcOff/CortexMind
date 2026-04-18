//
// Created by muham on 18.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h> // from _deps/googletest-src

#include "CortexMind/framework/Memory/forge.hpp"
#include "CortexMind/framework/Memory/transform.hpp"
#include "CortexMind/core/Tools/utils.cuh"
#include "vector"

using namespace cortex::_fw::sys;
using namespace cortex::_fw;

TEST(ForgeChunkTest, DeviceWriteRead) {
    ForgeChunk forge(1024);
    f32* d_ptr = forge.allocate(10, 256);
    ASSERT_NE(d_ptr, nullptr);

    const std::vector h_src(10, 3.14f);
    std::vector h_dst(10, 0.0f);

    transform<f32>::upload(d_ptr, h_src.data(), 10);
    transform<f32>::download(h_dst.data(), d_ptr, 10);

    for(int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(h_dst[i], 3.14f);
    }
}

TEST(ForgeChunkTest, CudaAlignment256) {
    ForgeChunk forge(2048);
    static_cast<void>(forge.allocate(7));

    f32* d_ptr = forge.allocate(100, 256);
    const auto addr = reinterpret_cast<uintptr_t>(d_ptr);

    EXPECT_EQ(addr % 256, 0);
}

TEST(ForgeChunkTest, BulkResetTest) {
    ForgeChunk forge(5000);

    std::vector<f32*> ptrs;
    for(int i = 0; i < 10; ++i) {
        ptrs.push_back(forge.allocate(100));
    }

    EXPECT_EQ(forge.used(), 1000);
    forge.reset();
    EXPECT_EQ(forge.used(), 0);

    EXPECT_NE(forge.allocate(5000), nullptr);
}