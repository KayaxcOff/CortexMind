//
// Created by muham on 9.05.2026.
//

#include "CortexMind/framework/Tools/err.hpp"
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Tools/cuda.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/runtime/macros.hpp>
#include <cstdio>

using namespace cortex::_fw;

namespace {
    /**
     * @brief Trims a file path by returning a pointer to the last occurrence
     *        of the substring "CortexMind".
     *
     * This function scans the given null-terminated character string and updates
     * the returned pointer whenever the substring "CortexMind" is found.
     * If the substring exists multiple times, the pointer to the last occurrence
     * is returned.
     *
     * If the substring is not found, the original input pointer is returned.
     *
     * @param path Pointer to a null-terminated path string.
     * @return Pointer to the last occurrence of "CortexMind" within the input
     *         string, or the original pointer if no occurrence is found.
     *
     * @note The returned pointer refers to memory owned by the input string.
     *       No allocation or copying is performed.
     *
     * @warning The input string must remain valid for the lifetime of the
     *          returned pointer.
     */
    constexpr const char* trim_path(const char* path) noexcept {
        const char* output = path;
        const char* p = path;

        while (*p != '\0') {
            if (p[0] == 'C' && p[1] == 'o' && p[2] == 'r' && p[3] == 't' && p[4] == 'e' && p[5] == 'x' && p[6] == 'M' && p[7] == 'i' && p[8] == 'n' && p[9] == 'd') {
                output = p;
            }
            ++p;
        }
        return output;
    }
} //unnamed namespace

void err::exitIf(const bool condition, const std::string_view &message, const char *file, const i32 line) {
    if (condition) [[unlikely]] {
        std::fprintf(stderr, "[ERROR] [%s | %d] %s\n", trim_path(file), line, message.data());
        std::exit(CXM_ERR_EXIT);
    }
}

void err::warnIf(const bool condition, const std::string_view &message, const char *file, const i32 line) {
    if (condition) [[unlikely]] {
        std::fprintf(stderr, "[WARN]  [%s | %d] %s\n", trim_path(file), line, message.data());
    }
}

#if CXM_IS_CUDA_AVAILABLE
    void err::exitIf(const cudaError_t call, const char *file, const i32 line) {
        if (call != cudaSuccess) [[unlikely]] {
            std::fprintf(stderr, "[ERROR] [%s | %d] — CUDA: %s\n", file, line, cuda::ErrorAsString(call));
            std::exit(CXM_ERR_EXIT);
        }
    }
#endif //#if CXM_IS_CUDA_AVAILABLE