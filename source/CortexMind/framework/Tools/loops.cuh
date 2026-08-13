//
// Created by muham on 7.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_LOOPS_CUH
#define CORTEXMIND_FRAMEWORK_TOOLS_LOOPS_CUH

#include <device_launch_parameters.h>

/**
 * @brief Iterates over a one-dimensional range using all CUDA threads in the grid.
 *
 * Each thread starts at the global thread index and advances by the total
 * number of threads in the grid. This grid-stride loop allows the same kernel
 * to process arrays larger than the number of simultaneously launched threads.
 *
 * @param i Loop index variable.
 * @param N Number of elements in the iteration range.
 */
#define CXM_KERNEL_LOOP_1D(i, N)                                \
    for(std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;  \
    i < (N);                                                    \
    i += blockDim.x * gridDim.x)

/**
 * @brief Iterates over the tail of a one-dimensional range using CUDA threads.
 *
 * Similar to CXM_KERNEL_LOOP_1D, but starts iteration at the specified
 * tail offset. This is useful when the main vectorized or tiled portion
 * of an operation has already been processed and the remaining elements
 * need to be handled by a CUDA kernel.
 *
 * @param i Loop index variable.
 * @param tail_start Starting offset of the remaining range.
 * @param N Exclusive upper bound of the iteration range.
 */
#define CXM_KERNEL_LOOP_TAIL(i, tail_start, N)                          \
    for (std::size_t i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; \
    i < (N);                                                            \
    i += blockDim.x * gridDim.x)

#endif //CORTEXMIND_FRAMEWORK_TOOLS_LOOPS_CUH