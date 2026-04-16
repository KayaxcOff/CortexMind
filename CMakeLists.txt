cmake_minimum_required(VERSION 4.1)
project(CortexMind LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 17)

include(cmake/Dependencies.cmake)
include(cmake/CUDA.cmake)
include(cmake/targets.cmake)
include(cmake/tests.cmake)
include(cmake/compiler.cmake)