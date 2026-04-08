//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_LOOP_H
#define CORTEXMIND_CORE_TOOLS_LOOP_H

/**
 * @brief Macro for 1D strided loop across all threads in a CUDA kernel.
 *
 * This macro automatically calculates the global thread index and iterates
 * with a stride equal to the total number of threads in the grid.
 * It ensures every element in the array is processed exactly once, even when
 * the number of elements is larger than the total number of launched threads.
 *
 * Usage example:
 * @code
 * CXM_CUDA_LOOP_1D(i, N) {
 *     // process element i
 * }
 * @endcode
 *
 * @param i Loop variable (usually `size_t i`)
 * @param N Total number of elements to process
 */
#define CXM_CUDA_LOOP_1D(i, N)                                      \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;          \
    i < (N);                                                        \
    i += blockDim.x * gridDim.x)

/**
 * @brief Macro for handling the tail (remaining) elements in CUDA kernels.
 *
 * This macro is used when the total number of elements is not a multiple of the
 * vector width (e.g. after processing float4 chunks). It starts from `tail_start`
 * and uses strided iteration so that all remaining elements are processed exactly once.
 *
 * @param i           Loop variable (usually `size_t i`)
 * @param tail_start  Starting index of the remaining elements
 * @param N           Total number of elements
 */
#define CXM_CUDA_LOOP_TAIL(i, tail_start, N)                            \
    for (size_t i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; \
    i < (N);                                                            \
    i += blockDim.x * gridDim.x)

#endif //CORTEXMIND_CORE_TOOLS_LOOP_H