include(FetchContent)

# =============================================================================
# 1. CORE LIBRARIES (Dependencies to be compiled every time)
# =============================================================================

# nlohmann_json
FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.zip
)

# TLX
FetchContent_Declare(
        TLX
        GIT_REPOSITORY https://github.com/KayaxcOff/TLX.git
        GIT_TAG main
)

FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master
)

set(CXM_CORE_DEPS nlohmann_json TLX stb)

# =============================================================================
# 2. OPTIONAL TESTING LIBRARIES (Will be built only if BUILD_TESTING=ON)
# =============================================================================
set(CXM_TEST_DEPS "")

if(BUILD_TESTING)
    # We are forcibly disabling Google Benchmark's own tests.
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

    # GoogleTest settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # Prevents runtime conflicts on Windows/MSVC.

    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    )

    FetchContent_Declare(
            benchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.9.4
    )

    list(APPEND CXM_TEST_DEPS googletest benchmark)
endif()

# =============================================================================
# 4. THE CYCLE OF ACTIVATING ADDICTIONS
# =============================================================================
foreach(dep IN LISTS CXM_CORE_DEPS CXM_TEST_DEPS)
    message(STATUS "CXM: Fetching dependency -> ${dep}")
    FetchContent_MakeAvailable(${dep})
endforeach()