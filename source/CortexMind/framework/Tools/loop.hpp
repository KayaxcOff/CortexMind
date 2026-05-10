//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_LOOP_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_LOOP_HPP

#include <device_launch_parameters.h>

/**
 * @def CXM_CUDA_LOOP_1D(i, N)
 * @brief Standard grid-stride 1D loop for CUDA kernels.
 *
 * Iterates over a 1D range `[0, N)` using a grid-stride pattern.
 * Each thread processes elements with a stride equal to the total number
 * of threads in the grid.
 *
 * @param i Loop variable (usually `size_t i`)
 * @param N Total number of elements to process
 *
 * @code
 * __global__ void myKernel(const float* x, float* y, size_t N) {
 *     CXM_CUDA_LOOP_1D(i, N) {
 *         y[i] = x[i] * 2.0f;
 *     }
 * }
 * @endcode
 */
#define CXM_CUDA_LOOP_1D(i, N) \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    i < (N); \
    i += blockDim.x * gridDim.x)

/**
 * @def CXM_CUDA_LOOP_TAIL(i, tail_start, N)
 * @brief Grid-stride loop for processing tail/remainder elements.
 *
 * Used when you have already processed a main block and want to continue
 * with the remaining elements using the same grid-stride pattern.
 *
 * @param i          Loop variable
 * @param tail_start Starting index for the tail section
 * @param N          Total number of elements
 *
 * @code
 * size_t main_end = (N / 8) * 8;
 * CXM_CUDA_LOOP_1D(i, main_end) { ... }   // vectorized main loop
 * CXM_CUDA_LOOP_TAIL(i, main_end, N) {    // scalar tail
 *     // process remaining elements
 * }
 * @endcode
 */
#define CXM_CUDA_LOOP_TAIL(i, tail_start, N) \
    for (size_t i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; \
    i < (N);                                                            \
    i += blockDim.x * gridDim.x)

/**
 * @def CXM_CUDA_LOOP_1D_VEC4(i, N)
 * @brief Grid-stride 1D loop optimized for 4-element vectorized operations.
 *
 * Loops until `N/4` (integer division). Intended to be used when processing
 * 4 elements per iteration (e.g. `float4`, manual unrolling, or AVX-like vector ops).
 *
 * @param i Loop index variable (represents vector index, not element index)
 * @param N Total number of elements (not vectors)
 *
 * @note Equivalent to looping `i < (N >> 2)`
 */
#define CXM_CUDA_LOOP_1D_VEC4(i, N) \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    i < ((N) >> 2); \
    i += blockDim.x * gridDim.x)

#endif //CORTEXMIND_FRAMEWORK_TOOLS_LOOP_HPP