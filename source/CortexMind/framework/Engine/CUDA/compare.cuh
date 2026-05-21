//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_COMPARE_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_COMPARE_CUH

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::cuda {
    struct CompareTo {
        static void greater(const f32* Xx, const f32* Yy, f32* Xz, size_t N);
        static void less(const f32* Xx, const f32* Yy, f32* Xz, size_t N);
        static void greater_eq(const f32* Xx, const f32* Yy, f32* Xz, size_t N);
        static void less_eq(const f32* Xx, const f32* Yy, f32* Xz, size_t N);
        static bool equal(const f32* Xx, const f32* Yy, size_t N); // reduce → bool
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_COMPARE_CUH