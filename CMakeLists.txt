cmake_minimum_required(VERSION 4.1)
project(CortexMind)

set(CMAKE_CXX_STANDARD 20)

include(CTest)

include(cmake/cuda.cmake)
include(cmake/dependencies.cmake)
include(cmake/targets.cmake)
include(cmake/compile_flags.cmake)
if(BUILD_TESTING)
    include(cmake/tests.cmake)
endif()