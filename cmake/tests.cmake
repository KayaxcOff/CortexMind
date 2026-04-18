enable_testing()

add_executable(CXM_TEST
        test/main_test.cpp
)

target_link_libraries(CXM_TEST PRIVATE
        CortexMind
)


add_executable(CXM_G_TEST
        test/g_test.cpp
)

target_link_libraries(CXM_G_TEST PRIVATE
        CortexMind
        GTest::gtest
        GTest::gtest_main
)


add_executable(CXM_CUDA_G_TEST
        test/cuda_g_test.cpp
)

target_include_directories(CXM_CUDA_G_TEST PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(CXM_CUDA_G_TEST PRIVATE
        CortexMind
        GTest::gtest
        GTest::gtest_main
        CUDA::cudart
)

include(GoogleTest)
gtest_discover_tests(CXM_G_TEST)
gtest_discover_tests(CXM_CUDA_G_TEST)