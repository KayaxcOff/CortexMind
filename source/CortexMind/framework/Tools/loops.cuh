//
// Created by muham on 7.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_LOOPS_CUH
#define CORTEXMIND_FRAMEWORK_TOOLS_LOOPS_CUH

#include <device_launch_parameters.h>

#define CXM_KERNEL_LOOP_1D(i, N)                                \
    for(std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;  \
    i < (N);                                                    \
    i += blockDim.x * gridDim.x)

#define CXM_KERNEL_LOOP_TAIL(i, tail_start, N)                          \
    for (std::size_t i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; \
    i < (N);                                                            \
    i += blockDim.x * gridDim.x)

#endif //CORTEXMIND_FRAMEWORK_TOOLS_LOOPS_CUH