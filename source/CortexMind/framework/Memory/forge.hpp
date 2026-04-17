//
// Created by muham on 17.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_FORGE_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_FORGE_HPP

#include <CortexMind/runtime/macros.hpp>

namespace cortex::_fw::sys {
    class ForgeChunk {
    public:
        explicit ForgeChunk(size_t capacity = CXM_DEFAULT_POOL_SIZE);
    };
} //namespace cortex::_fw::sys

#endif //CORTEXMIND_FRAMEWORK_MEMORY_FORGE_HPP