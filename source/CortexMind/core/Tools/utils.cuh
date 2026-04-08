//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_UTILS_CUH
#define CORTEXMIND_CORE_TOOLS_UTILS_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/framework/Tools/params.hpp>
#include <CortexMind/framework/Tools/err.hpp>

namespace cortex::_fw::cuda {
    constexpr cudaMemcpyKind CXM_HOST_TO_DEVICE     = cudaMemcpyHostToDevice;
    constexpr cudaMemcpyKind CXM_DEVICE_TO_HOST     = cudaMemcpyDeviceToHost;
    constexpr cudaMemcpyKind CXM_DEVICE_TO_DEVICE   = cudaMemcpyDeviceToDevice;

    inline constexpr i32 WARP_SIZE = 32;
    inline constexpr i32 BLOCK_SIZE_1D = 256;
    inline constexpr i32 BLOCK_SIZE_2D = 16;
    inline constexpr i32 MAT_TILE = 16;

    /**
     * @brief Calculates 1D grid dimensions for kernel launch.
     * @param n     Number of elements to process
     * @param block Block size (default: 256)
     * @return dim3 grid configuration
     */
    [[nodiscard]]
    __host__ inline dim3 grid1d(const size_t n, const i32 block = BLOCK_SIZE_1D) {
        return dim3(static_cast<unsigned>((n + block - 1) / block));
    }
    /**
     * @brief Calculates 2D grid dimensions for kernel launch.
     * @param rows  Number of rows
     * @param cols  Number of columns
     * @param tile  Tile size (default: 16)
     * @return dim3 grid configuration (x = cols, y = rows)
     */
    [[nodiscard]]
    __host__ inline dim3 grid2d(const size_t rows, const size_t cols, const i32 tile = BLOCK_SIZE_2D) {
        return dim3(static_cast<unsigned>((cols + tile - 1) / tile), static_cast<unsigned>((rows + tile - 1) / tile));
    }
    /**
     * @brief Returns the global thread index in a 1D kernel.
     * @return Global thread ID
     */
    [[nodiscard]]
    __device__ inline size_t global_thread_id() {
        return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    }
    /**
     * @brief Rounds up a value to the nearest multiple of alignment.
     * @param n     Value to align
     * @param align Alignment size
     * @return Aligned value
     */
    [[nodiscard]]
    __host__ __device__ constexpr size_t round_up(const size_t n, const size_t align) {
        return (n + align - 1) / align * align;
    }

    /**
     * @brief Simplified CUDA memory copy wrapper.
     * @param src  Source pointer
     * @param dst  Destination pointer
     * @param size Size in bytes
     * @param kind Direction of copy (HostToDevice, DeviceToHost, etc.)
     */
    inline void memcpy(const void* src, void* dst, const size_t size, const cudaMemcpyKind kind) {
        CXM_CUDA_ASSERT(cudaMemcpy(dst, src, size, kind), "cortex::_fw::cuda::memcpy()");
    }

    /**
     * @brief Synchronizes the device with the host.
     *
     * This function blocks the host thread until all previously issued CUDA
     * commands on the device have completed. It is typically used for debugging
     * or ensuring that GPU computations are finished before proceeding.
     */
    inline void DeviceSynchronize() {
        CXM_CUDA_ASSERT(cudaDeviceSynchronize(), "cortex::_fw::cuda::DeviceSynchronize()");
    }

    /**
     * @brief Checks for the last CUDA error.
     *
     * This function retrieves the last error that has occurred on the CUDA device.
     * It is commonly used after kernel launches or CUDA API calls to ensure that
     * no asynchronous errors have occurred during execution.
     */
    inline void GetLastError() {
        CXM_CUDA_ASSERT(cudaGetLastError(), "cortex::_fw::cuda::GetLastError()");
    }

    /**
     * @brief Returns a human-readable string for a given CUDA error code.
     *
     * Simple wrapper around `cudaGetErrorString()` for cleaner error reporting.
     *
     * @param call CUDA error code returned by a CUDA API function
     * @return Const char pointer to the error description string
     */
    inline const char* ErrorAsString(cudaError_t call) {
        return cudaGetErrorString(call);
    }
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_TOOLS_UTILS_CUH