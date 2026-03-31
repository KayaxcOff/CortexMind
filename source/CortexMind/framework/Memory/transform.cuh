//
// Created by muham on 31.03.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_TRANSFORM_CUH
#define CORTEXMIND_FRAMEWORK_MEMORY_TRANSFORM_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/framework/Tools/err.hpp>
#include <cstring>

namespace cortex::_fw::sys {
    template<typename T>
    struct transform {
        static void upload(const T* src, T* dst, const size_t N) {
            if (N == 0) {
                return;
            }

            CXM_ASSERT(
                cudaMemcpy(dst, src, N * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess,
                "Host to Device transfer failed",
                "cortex::_fw::sys::transform<>::upload()"
            );
        }

        static void download(const T* src, T* dst, const size_t N) {
            if (N == 0) {
                return;
            }

            CXM_ASSERT(
                cudaMemcpy(dst, src, N * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess,
                "Device to Host transfer failed",
                "cortex::_fw::sys::transform::download"
            );
        }

        static void copy_h2h(const T* src, T* dst, const size_t N) {
            if (N == 0) {
                return;
            }

            std::memcpy(dst, src, N * sizeof(T));
        }

        static void copy_d2d(T* src, T* dst, const size_t N) {
            if (N == 0) {
                return;
            }

            CXM_ASSERT(
                cudaMemcpy(dst, src, N * sizeof(T), cudaMemcpyDeviceToDevice) == cudaSuccess,
                "Device to Device transfer failed",
                "cortex::_fw::sys::transform::copy_d2d"
            );
        }

        static void upload(const T& src, T* dst) {
            CXM_ASSERT(
                cudaMemcpy(dst, &src, sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess,
                "Host to Device scalar transfer failed",
                "cortex::_fw::sys::transform::upload (scalar)"
            );
        }

        static void download(const T* src, T& dst) {
            CXM_ASSERT(
                cudaMemcpy(&dst, src, sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess,
                "Device to Host scalar transfer failed",
                "cortex::_fw::sys::transform::download (scalar)"
            );
        }

        static void upload(const T* src, T* dst, const size_t N, cudaStream_t stream) {
            if (N == 0) return;
            CXM_ASSERT(
                cudaMemcpyAsync(dst, src, N * sizeof(T), cudaMemcpyHostToDevice, stream) == cudaSuccess,
                "Host to Device async transfer failed",
                "cortex::_fw::sys::transform::upload_async"
            );
        }

        static void download(const T* src, T* dst, const size_t N, cudaStream_t stream) {
            if (N == 0) return;
            CXM_ASSERT(
                cudaMemcpyAsync(dst, src, N * sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess,
                "Device to Host async transfer failed",
                "cortex::_fw::sys::transform::download_async"
            );
        }

        static void copy_d2d(const T* src, T* dst, const size_t N, cudaStream_t stream) {
            if (N == 0) return;
            CXM_ASSERT(
                cudaMemcpyAsync(dst, src, N * sizeof(T), cudaMemcpyDeviceToDevice, stream) == cudaSuccess,
                "Device to Device async transfer failed",
                "cortex::_fw::sys::transform::copy_d2d_async"
            );
        }

        static void upload(const T& src, T* dst, cudaStream_t stream) {
            CXM_ASSERT(
                cudaMemcpyAsync(dst, &src, sizeof(T), cudaMemcpyHostToDevice, stream) == cudaSuccess,
                "Host to Device scalar async transfer failed",
                "cortex::_fw::sys::transform::upload_async (scalar)"
            );
        }

        static void download(const T* src, T& dst, cudaStream_t stream) {
            CXM_ASSERT(
                cudaMemcpyAsync(&dst, src, sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess,
                "Device to Host scalar async transfer failed",
                "cortex::_fw::sys::transform::download_async (scalar)"
            );
        }
    };
} //namespace cortex::_fw::sys

#endif //CORTEXMIND_FRAMEWORK_MEMORY_TRANSFORM_CUH