//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_ALLOCATORS_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_ALLOCATORS_HPP

#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Memory/forge.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Memory/mem.hpp>
#include <CortexMind/runtime/macros.hpp>

namespace cortex::_fw {
    #if CXM_IS_CUDA_AVAILABLE
        inline ForgeChunk forge(CXM_DEFAULT_POOL_SIZE);
    #endif //#if CXM_IS_CUDA_AVAILABLE

    inline TrackedMem mem(CXM_DEFAULT_POOL_SIZE);
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_MEMORY_ALLOCATORS_HPP