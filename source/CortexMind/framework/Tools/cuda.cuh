//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_CUDA_CUH
#define CORTEXMIND_FRAMEWORK_TOOLS_CUDA_CUH

#include <CortexMind/framework/Tools/types.hpp>
#include <cuda_runtime.h>

namespace cortex::_fw::cuda {
    inline constexpr cudaMemcpyKind CXM_HOST_TO_DEVICE     = cudaMemcpyHostToDevice;
    inline constexpr cudaMemcpyKind CXM_DEVICE_TO_HOST     = cudaMemcpyDeviceToHost;
    inline constexpr cudaMemcpyKind CXM_DEVICE_TO_DEVICE   = cudaMemcpyDeviceToDevice;

    constexpr unsigned int CXM_HOST_ALLOC_DEFAULT        = cudaHostAllocDefault;
    constexpr unsigned int CXM_HOST_ALLOC_PORTABLE       = cudaHostAllocPortable;
    constexpr unsigned int CXM_HOST_ALLOC_MAPPED         = cudaHostAllocMapped;
    constexpr unsigned int CXM_HOST_ALLOC_WRITE_COMBINED = cudaHostAllocWriteCombined;

    inline constexpr i32 WARP_SIZE      = 32;
    inline constexpr i32 BLOCK_SIZE_1D  = 256;
    inline constexpr i32 BLOCK_SIZE_2D  = 16;
    inline constexpr i32 MAT_TILE       = 16;

    /**
     * @brief Calculates 1D grid dimensions for CUDA kernels.
     *
     * @param n     Total number of elements to process
     * @param block Block size (default = BLOCK_SIZE_1D)
     * @return dim3 grid configuration
     */
    [[nodiscard]]
    __host__ dim3 grid1d(size_t n, i32 block = BLOCK_SIZE_1D);
    /**
     * @brief Calculates 2D grid dimensions for CUDA kernels.
     *
     * @param rows  Number of rows
     * @param cols  Number of columns
     * @param tile  Tile size (default = BLOCK_SIZE_2D)
     * @return dim3 grid configuration (x = cols, y = rows)
     */
    [[nodiscard]]
    __host__ dim3 grid2d(const size_t rows, const size_t cols, const i32 tile = BLOCK_SIZE_2D);
    /**
     * @brief Wrapper for cudaMemcpy with error checking.
     */
    void memcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind);

    /**
     * @brief Synchronizes the current CUDA device.
     */
    void DeviceSynchronize();
    /**
     * @brief Checks the last CUDA error.
     */
    void GetLastError();
    /**
     * @brief Returns a human-readable string for a CUDA error code.
     */
    const char* ErrorAsString(cudaError_t call);
    /**
     * @brief Typed wrapper for cudaMemset.
     *
     * @tparam T Type of the elements
     * @param ptr   Device pointer
     * @param value Value to set
     * @param size  Number of elements (not bytes)
     */
    template<typename T>
    [[nodiscard]]
    inline cudaError_t memset(T* ptr, T value, const size_t size) {
        return cudaMemset(ptr, value, sizeof(T) * size);
    }
    /**
     * @brief Wrapper for cudaMalloc with error checking.
     */
    void malloc(void** ptr, size_t size);
    /**
     * @brief Wrapper for cudaFree with error checking.
     */
    void free(void* ptr);
    /**
     * @brief Host memory management utilities.
     */
    struct host {
        /**
         * @brief Allocates page-locked (pinned) host memory.
         *
         * @param ptr   Pointer to be filled with allocated address
         * @param size  Size in bytes
         * @param flags Allocation flags (default = CXM_HOST_ALLOC_DEFAULT)
         */
        static void allocate(void** ptr, size_t size, unsigned int flags = CXM_HOST_ALLOC_DEFAULT);

        /**
         * @brief Frees host memory allocated with cudaHostAlloc.
         */
        static void free(void* ptr);
    };
    /**
     * @brief Maps pinned host memory to device address space.
     */
    void map(void** device_ptr, void* host_ptr);

    /**
     * @brief CUDA warp-level shuffle (shfl) operations.
     *
     * Convenient wrapper around `__shfl_*_sync` intrinsics.
     */
    struct shfl {
        /**
         * @brief Shuffle down within a warp.
         */
        template<typename T>
        __device__ __forceinline__ static T down(T val, int offset, unsigned mask=0xFFFFFFFF) {
            return __shfl_down_sync(mask, val, offset);
        }
        /**
         * @brief Shuffle up within a warp.
         */
        template<typename T>
        __device__ __forceinline__ static T up(T val, int offset, unsigned mask=0xFFFFFFFF) {
            return __shfl_up_sync(mask, val, offset);
        }
        /**
         * @brief Shuffle with XOR mask (useful for reductions, butterfly patterns etc.).
         */
        template<typename T>
        __device__ __forceinline__ static T xor_op(T val, int laneMask, unsigned mask=0xFFFFFFFF) {
            return __shfl_xor_sync(mask, val, laneMask);
        }
    };

    #ifdef __CUDACC__
        /**
         * @brief Atomic operations wrapper.
         *
         * Provides type-safe atomic functions for common types.
         */
        struct atomic {
            template<typename T>
            __device__ __forceinline__ static void add(T* addr, T val) {
                ::atomicAdd(addr, val);  // f32 ve i32 için zaten overload var
            }
        };

        /**
         * @brief Synchronizes all threads in the current block.
         */
        __device__ inline void SynchronizeThreads() {
            __syncthreads();
        }

        /**
         * @brief Returns the global thread index in a 1D kernel.
         *
         * @return Global thread ID (0 to gridDim.x * blockDim.x - 1)
         */
        [[nodiscard]]
        __device__ inline size_t global_thread_id() {
            return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        }
    #endif //#ifdef __CUDACC__
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_FRAMEWORK_TOOLS_CUDA_CUH