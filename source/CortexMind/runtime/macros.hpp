//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_RUNTIME_MACROS_HPP
#define CORTEXMIND_RUNTIME_MACROS_HPP

#if CXM_IS_CUDA_AVAILABLE
#include <cuda_runtime.h>
    #define CXM_GLOBAL __global__
    #define CXM_SHARED __shared__
    #define CXM_DEVICE_ATTR __device__ __host__
    #define CXM_DEVICE __device__
    #define CXM_HOST __host__
    #define CXM_DEVICE_INLINE __inline__
#else //#if CXM_IS_CUDA_AVAILABLE
    #define CXM_GLOBAL
    #define CXM_SHARED
    #define CXM_DEVICE_ATTR
    #define CXM_DEVICE
    #define CXM_HOST
    #define CXM_DEVICE_INLINE
#endif //#if CXM_IS_CUDA_AVAILABLE #else

#define CXM_SAFETY_EXIT 0
#define CXM_ERR_EXIT    1

#endif //CORTEXMIND_RUNTIME_MACROS_HPP