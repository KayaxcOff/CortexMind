add_executable(CXM_MAIN_TEST tests/main.cpp)
add_executable(CXM_G_TEST tests/g.cpp)
add_executable(CXM_BENCHMARK tests/benchmark.cpp)

if(CXM_IS_CUDA_AVAILABLE)
    add_executable(CXM_CUDA_G_TEST tests/main.cu)
endif()

set(CXM_ALL_TEST_EXECUTABLES CXM_MAIN_TEST CXM_G_TEST CXM_BENCHMARK)
if(CXM_IS_CUDA_AVAILABLE)
    list(APPEND CXM_ALL_TEST_EXECUTABLES CXM_CUDA_G_TEST)
endif()

foreach(target IN LISTS CXM_ALL_TEST_EXECUTABLES)
    target_link_libraries(${target} PRIVATE CortexMind)

    if(CXM_IS_CUDA_AVAILABLE)
        target_include_directories(${target} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(${target} PRIVATE CUDA::cudart)
    endif()
endforeach()

set(CXM_GTEST_TARGETS CXM_G_TEST)
if(CXM_IS_CUDA_AVAILABLE)
    list(APPEND CXM_GTEST_TARGETS CXM_CUDA_G_TEST)
endif()

foreach(gtest_target IN LISTS CXM_GTEST_TARGETS)
    target_link_libraries(${gtest_target} PRIVATE GTest::gtest GTest::gtest_main)
endforeach()

target_link_libraries(CXM_BENCHMARK PRIVATE benchmark::benchmark)

include(GoogleTest)
gtest_discover_tests(CXM_G_TEST)
if(CXM_IS_CUDA_AVAILABLE)
    gtest_discover_tests(CXM_CUDA_G_TEST)
endif()

add_test(NAME CXM_MAIN_TEST COMMAND CXM_MAIN_TEST)
add_test(NAME CXM_G_TEST COMMAND CXM_G_TEST)
if(CXM_IS_CUDA_AVAILABLE)
    add_test(NAME CXM_CUDA_G_TEST COMMAND CXM_CUDA_G_TEST)
endif()