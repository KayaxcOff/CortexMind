//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_UTILS_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_UTILS_CUH

#include <CortexMind/core/Engine/CUDA/params.cuh>
#include <CortexMind/core/Tools/defaults.hpp>
#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::cuda {
     /**
      * @name   Warp & Block Size Constants
      * @brief  Standard values used in most kernels
      */
     //@{
    inline constexpr int WARP_SIZE      = 32;   ///< Number of threads in a warp (hardware constant)
    inline constexpr int BLOCK_SIZE_1D  = 256;  ///< Default 1D block size (multiple of warp size)
    inline constexpr int BLOCK_SIZE_2D  = 16;   ///< Default tile side length for 2D blocks (16×16 = 256 threads)
    //@}

    /**
     * @name   Matrix Tiling Parameter
     * @brief  Tile size used in tiled matrix operations (GEMM-like kernels)
     */
    //@{
    inline constexpr int MAT_TILE = 16;  ///< Default tile size for shared memory tiling in matmul kernels
    //@}

    /**
     * @brief   Rounds up n to the next multiple of align
     * @param   n       Value to be aligned
     * @param   align   Alignment requirement (must be power of 2 in most cases)
     * @return  Smallest value ≥ n that is multiple of align
     *
     * @note    Commonly used for padding arrays to shared memory bank boundaries
     *          or for grid/block calculations.
     */
    __host__ __device__
    inline constexpr size_t round_up(size_t n, size_t align) {
        return (n + align - 1) / align * align;
    }
    /**
     * @brief   Computes 1D grid dimension for a given number of elements
     * @param   n       Total number of elements / threads to launch
     * @param   block   Block size (default = BLOCK_SIZE_1D)
     * @return  dim3 with x = ceil(n / block), y=z=1
     *
     * @example
     *   kernel<<<grid1d(N), BLOCK_SIZE_1D>>>(...);
     */
    __host__
    inline dim3 grid1d(size_t n, int block = BLOCK_SIZE_1D) {
        return dim3(static_cast<unsigned>((n + block - 1) / block));
    }
    /**
     * @brief   Computes 2D grid dimension for a 2D problem (rows × cols)
     * @param   rows    Number of rows / height
     * @param   cols    Number of columns / width
     * @param   tile    Tile size per block dimension (default = BLOCK_SIZE_2D)
     * @return  dim3 with x = ceil(cols / tile), y = ceil(rows / tile), z=1
     *
     * @example
     *   kernel_2d<<<grid2d(M, N), dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D)>>>(...);
     */
    __host__
    inline dim3 grid2d(size_t rows, size_t cols, int tile = BLOCK_SIZE_2D) {
        return dim3(
            static_cast<unsigned>((cols + tile - 1) / tile),
            static_cast<unsigned>((rows + tile - 1) / tile)
        );
    }

    /**
     * @brief   Returns the globally unique 1D thread index across the entire grid
     * @return  Linear global thread ID = blockIdx.x * blockDim.x + threadIdx.x
     *
     * @note    Only valid in 1D grid launches (gridDim.y = gridDim.z = 1)
     * @note    Equivalent to: blockIdx.x * blockDim.x + threadIdx.x
     * @note    Useful in strided loops or when flattening multi-dimensional work
     */
    __device__ inline size_t global_thread_id() {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    /**
     * @def     CXM_CUDA_LOOP(i, idx)
     * @brief   Strided 1D loop across all threads in the grid
     * @param   i       Loop variable name (will be declared as size_t)
     * @param   idx     Total number of elements / iterations to process
     *
     * This macro creates a loop that:
     *   - starts from the current thread's global index
     *   - increments by the total number of threads in the grid (grid stride)
     *   - continues until i >= idx
     *
     * Typical usage (inside kernel):
     * @code
     *     CXM_CUDA_LOOP(i, N) {
     *         // process element i
     *         output[i] = input[i] * 2.0f;
     *     }
     * @endcode
     *
     * @note    Assumes 1D grid launch (blockIdx.y/z and gridDim.y/z ignored)
     * @note    Very common pattern for large 1D arrays or flattened tensors
     * @note    Helps avoid warp divergence and improves memory access coalescing
     */
    #define CXM_CUDA_LOOP(i, idx) \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < (idx); \
             i += blockDim.x * gridDim.x)

    #ifdef CXM_DEBUG
    #define CXM_CUDA_ASSERT(kernel_call) do {           \
        auto err = (kernel_call); \
        cudaDeviceSynchronize();                         \
        cudaError_t err = cudaGetLastError();            \
        if (err != cudaSuccess) {                        \
            fprintf(stderr, "[CortexMind] CUDA error at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::exit(CXM_ERR_EXIT);                     \
        }                                                \
    } while(0)
    #else
    #define CXM_CUDA_ASSERT(kernel_call) kernel_call
    #endif

    #ifdef CXM_DEBUG
    #define CXM_CUDA_CHECK() do { \
        cudaDeviceSynchronize(); \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[CortexMind] CUDA error at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::exit(CXM_ERR_EXIT); \
        } \
    } while(0)
    #else
    #define CXM_CUDA_CHECK() ((void)0)
    #endif
} // namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_UTILS_CUH