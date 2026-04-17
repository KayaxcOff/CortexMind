//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h> // from _deps/googletest-src

#include "CortexMind/framework/Memory/mem.hpp"

using namespace cortex::_fw::sys;
using namespace cortex::_fw;

TEST(TrackedMemTest, BasicAllocation) {
    TrackedMem mem(100);

    const auto ptr = mem.allocate(10);

    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(mem.used(), 10);
    EXPECT_EQ(mem.capacity(), 100);
}

TEST(TrackedMemTest, AllocationFailsWhenFull) {
    TrackedMem mem(10);

    const auto p1 = mem.allocate(10);
    const auto p2 = mem.allocate(1);

    EXPECT_NE(p1, nullptr);
    EXPECT_EQ(p2, nullptr);
}

TEST(TrackedMemTest, DeallocateWorks) {
    TrackedMem mem(50);

    const auto ptr = mem.allocate(20);
    ASSERT_NE(ptr, nullptr);

    mem.deallocate(ptr);

    EXPECT_EQ(mem.used(), 0);
}

TEST(TrackedMemTest, AlignmentGuaranteed) {
    TrackedMem mem(100);
    auto it = mem.allocate(10);

    const auto ptr = mem.allocate(10, 32);
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % 32, 0);
}

TEST(TrackedMemTest, TripleCloalesce) {
    TrackedMem mem(100);
    const auto p1 = mem.allocate(10);
    const auto p2 = mem.allocate(10);
    const auto p3 = mem.allocate(10);

    mem.deallocate(p1);
    mem.deallocate(p3);
    mem.deallocate(p2);

    EXPECT_EQ(mem.used(), 0);
    const auto p_full = mem.allocate(100);
    EXPECT_NE(p_full, nullptr);
}

TEST(TrackedMemTest, MinimalPaddingCorrected) {
    TrackedMem mem(100);
    const auto p1 = mem.allocate(5);

    const auto p2 = mem.allocate(5, 4);

    EXPECT_EQ(p2, p1 + 5);
}

TEST(TrackedMemTest, StressTest) {
    TrackedMem mem(1000);
    std::vector<f32*> ptrs;

    for(int i = 0; i < 50; ++i) {
        ptrs.push_back(mem.allocate(10));
    }

    for(auto p : ptrs) {
        mem.deallocate(p);
    }

    EXPECT_EQ(mem.used(), 0);
    EXPECT_NE(mem.allocate(1000), nullptr);
}