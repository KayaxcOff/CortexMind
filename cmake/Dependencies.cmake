include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/archive/refs/heads/develop.zip
)

FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master
)

FetchContent_MakeAvailable(googletest nlohmann_json stb)