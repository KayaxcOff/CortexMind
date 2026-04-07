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

#endif //CORTEXMIND_CORE_TOOLS_LOOP_H