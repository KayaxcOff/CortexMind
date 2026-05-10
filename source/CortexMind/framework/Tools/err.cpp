//
// Created by muham on 9.05.2026.
//

#include "CortexMind/framework/Tools/err.hpp"
#include <CortexMind/framework/Tools/cuda.cuh>
#include <cstdio>

using namespace cortex::_fw;

namespace {
    constexpr const char* trim_path(const char* path) noexcept {
        const char* output = path;
        const char* p = path;

        while (*p != '\0') {
            if (p[0] == 'C' && p[1] == 'o' && p[2] == 'r' &&
                p[3] == 't' && p[4] == 'e' && p[5] == 'x' &&
                p[6] == 'M' && p[7] == 'i' && p[8] == 'n' && p[9] == 'd') {
                output = p;
                }
            ++p;
        }
        return output;
    }

    [[noreturn]]
    __forceinline void exit_impl(const char* message, const char* file, const int line) noexcept {
        std::fprintf(stderr, "[ERROR] [%s | %d] %s\n", file, line, message);
        std::abort();
    }

    __forceinline void warn_impl(const char* message, const char* file, const int line) noexcept {
        std::fprintf(stderr, "[WARN]  [%s | %d] %s\n", file, line, message);
    }

    #if CXM_IS_CUDA_AVAILABLE
        [[noreturn]]
        __forceinline void cuda_exit_impl(const cudaError_t code, const char* file, const int line) noexcept {
            std::fprintf(stderr, "[ERROR] [%s | %d] — CUDA: %s\n", file, line, cuda::ErrorAsString(code));
            std::abort();
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE
} //unnamed namespace

void err::exitIf(const bool condition, const std::string_view &message, const char *file, const i32 line) {
    if (condition) [[unlikely]] {
        exit_impl(message.data(), trim_path(file), line);
    }
}

void err::warnIf(const bool condition, const std::string_view &message, const char *file, const i32 line) {
    if (condition) [[unlikely]] {
        warn_impl(message.data(), trim_path(file), line);
    }
}

#if CXM_IS_CUDA_AVAILABLE
    void err::exitIf_c(const cudaError_t call, const char *file, const i32 line) {
        if (call != cudaSuccess) [[unlikely]] {
            cuda_exit_impl(call, trim_path(file), line);
        }
    }
#endif //#if CXM_IS_CUDA_AVAILABLE