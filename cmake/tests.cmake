enable_testing()

add_executable(CXM_TEST
        test/main_test.cpp
)
add_executable(CXM_G_TEST
        test/g_test.cpp
)
add_executable(CXM_CUDA_G_TEST
        test/cuda_g_test.cpp
)

target_link_libraries(CXM_TEST PRIVATE
        CortexMind
)

target_link_libraries(CXM_G_TEST PRIVATE
        CortexMind
        GTest::gtest
        GTest::gtest_main
)

target_link_libraries(CXM_CUDA_G_TEST PRIVATE
        CortexMind
        GTest::gtest
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(CXM_G_TEST)
gtest_discover_tests(CXM_CUDA_G_TEST)