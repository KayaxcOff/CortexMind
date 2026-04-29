//
// Created by muham on 29.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_BROADCAST_H
#define CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_BROADCAST_H

#include <CortexMind/core/Tools/broadcast.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    struct MatrixBroadcast {
        static void add(const f32* Xx, const f32* Xy, f32* Xz, size_t N, const BroadcastInfo* info_ptr = nullptr);
        static void sub(const f32* Xx, const f32* Xy, f32* Xz, size_t N, const BroadcastInfo* info_ptr = nullptr);
        static void mul(const f32* Xx, const f32* Xy, f32* Xz, size_t N, const BroadcastInfo* info_ptr = nullptr);
        static void div(const f32* Xx, const f32* Xy, f32* Xz, size_t N, const BroadcastInfo* info_ptr = nullptr);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_BROADCAST_H