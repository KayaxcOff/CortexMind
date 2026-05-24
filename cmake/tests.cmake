enable_testing()

add_executable(CXM_MAIN_TEST
        test/main_test.cpp
)

if(CXM_CUDA_AVAILABLE)
    target_include_directories(CXM_MAIN_TEST PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(CXM_MAIN_TEST PRIVATE CortexMind CUDA::cudart)
else()
    target_link_libraries(CXM_MAIN_TEST PRIVATE CortexMind)
endif()


add_executable(CXM_G_TEST
        test/g_test.cpp
)

if(CXM_CUDA_AVAILABLE)
    target_include_directories(CXM_G_TEST PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(CXM_G_TEST PRIVATE CortexMind GTest::gtest GTest::gtest_main CUDA::cudart)
else()
    target_link_libraries(CXM_G_TEST PRIVATE CortexMind GTest::gtest GTest::gtest_main)
endif()

if(CXM_CUDA_AVAILABLE)
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
    gtest_discover_tests(CXM_CUDA_G_TEST)
endif()

include(GoogleTest)
gtest_discover_tests(CXM_G_TEST)