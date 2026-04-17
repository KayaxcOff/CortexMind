//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <gtest/gtest.h> // from _deps/googletest-src

#include "CortexMind/framework/Memory/mem.hpp"

using namespace cortex::_fw::sys;

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