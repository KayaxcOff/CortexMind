//
// Created by muham on 29.12.2025.
//

#ifndef CORTEXMIND_CORE_ENGINE_MEMORY_BLOCKS_HPP
#define CORTEXMIND_CORE_ENGINE_MEMORY_BLOCKS_HPP

namespace cortex::_fw::sys {
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };

    struct PoolBlock {
        void* ptr;
        size_t size;
        size_t offset;
    };
} // namespace cortex::_fw::sys

#endif // CORTEXMIND_CORE_ENGINE_MEMORY_BLOCKS_HPP