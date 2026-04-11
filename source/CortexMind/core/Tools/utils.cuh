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

    constexpr unsigned int CXM_HOST_ALLOC_DEFAULT        = cudaHostAllocDefault;
    constexpr unsigned int CXM_HOST_ALLOC_PORTABLE       = cudaHostAllocPortable;
    constexpr unsigned int CXM_HOST_ALLOC_MAPPED         = cudaHostAllocMapped;
    constexpr unsigned int CXM_HOST_ALLOC_WRITE_COMBINED = cudaHostAllocWriteCombined;

    inline constexpr i32 WARP_SIZE      = 32;
    inline constexpr i32 BLOCK_SIZE_1D  = 256;
    inline constexpr i32 BLOCK_SIZE_2D  = 16;
    inline constexpr i32 MAT_TILE       = 16;

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
    #ifdef __CUDACC__
        /**
         * @brief Returns the global thread index in a 1D kernel.
         * @return Global thread ID
         */
        [[nodiscard]]
        __device__ inline size_t global_thread_id() {
            return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        }
    #endif //#ifdef __CUDACC__
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
    inline void memcpy(void* dst, const void* src, const size_t size, const cudaMemcpyKind kind) {
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

    /**
     * @brief Sets a block of device memory to a specified value.
     * @param ptr   Device pointer
     * @param value Value to set (usually 0)
     * @param size  Number of elements (not bytes)
     */
    template<typename T>
    inline void memset(T* ptr, T value, size_t size) {
        CXM_CUDA_ASSERT(cudaMemset(ptr, value, sizeof(T) * size), "cortex::_fw::cuda::memset()");
    }

    /**
     * @brief Host memory (pinned memory) management utilities.
     */
    struct host {
        /**
         * @brief Allocates page-locked (pinned) host memory.
         * @param ptr   Pointer to allocated memory
         * @param size  Size in bytes
         * @param flags Allocation flags (default: cudaHostAllocDefault)
         * @return cudaSuccess on success
         */
        [[nodiscard]]
        static cudaError_t allocate(void** ptr, size_t size, unsigned int flags = CXM_HOST_ALLOC_DEFAULT) {
            return cudaHostAlloc(ptr, size, flags);
        }

        /**
         * @brief Frees pinned host memory.
         * @param ptr Pointer to previously allocated pinned memory
         * @return cudaSuccess on success
         */
        [[nodiscard]]
        static cudaError_t free(void* ptr) {
            return cudaFreeHost(ptr);
        }
    };

    /**
     * @brief Maps pinned host memory to device address space (zero-copy memory).
     *
     * This function returns a device pointer that can be used directly in CUDA kernels
     * to access the same physical memory as the host pointer without explicit copying.
     * It is typically used together with `cudaHostAlloc(..., cudaHostAllocMapped)`.
     *
     * @param device_ptr  [out] Device pointer that can be used in kernels
     * @param host_ptr    Host pointer previously allocated with pinned memory
     * @return cudaSuccess on success
     */
    [[nodiscard]]
    inline cudaError_t map(void** device_ptr, void* host_ptr) {
        return cudaHostGetDevicePointer(device_ptr, host_ptr, 0);
    }

    /**
     * @brief Warp-level shuffle primitives (CUDA 9+).
     */
    struct shfl {
        template<typename T>
        __device__ __forceinline__ static T down(T val, int offset, unsigned mask=0xFFFFFFFF) {
            return __shfl_down_sync(mask, val, offset);
        }

        template<typename T>
        __device__ __forceinline__ static T up(T val, int offset, unsigned mask=0xFFFFFFFF) {
            return __shfl_up_sync(mask, val, offset);
        }

        template<typename T>
        __device__ __forceinline__ static T xor_op(T val, int laneMask, unsigned mask=0xFFFFFFFF) {
            return __shfl_xor_sync(mask, val, laneMask);
        }
    };

    #ifdef __CUDACC__
        /**
         * @brief Atomic operations wrapper.
         */
        struct atomic {
            /**
             * @brief Atomically adds a value to the address.
             * @tparam T Type of the value (f32 or i32 supported)
             */
            template<typename T>
            __device__ __forceinline__ static void add(T* addr, T val);

            template<>
            __device__ __forceinline__ static void add<f32>(f32* addr, f32 val) {
                ::atomicAdd(addr, val);
            }

            template<>
            __device__ __forceinline__ static void add<i32>(i32* addr, i32 val) {
                ::atomicAdd(addr, val);
            }
        };
    #endif //#ifdef __CUDACC__

    #ifdef __CUDACC__
        /**
         * @brief Synchronizes all threads within a CUDA block.
         *
         * Equivalent to `__syncthreads()`.
         */
        __device__ inline void SynchronizeThreads() {
            __syncthreads();
        }
    #endif //#ifdef __CUDACC__
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_TOOLS_UTILS_CUH