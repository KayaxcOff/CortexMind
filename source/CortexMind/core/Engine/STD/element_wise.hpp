//
// Created by muham on 26.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_STD_ELEMENT_WISE_HPP
#define CORTEXMIND_CORE_ENGINE_STD_ELEMENT_WISE_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::stl {
    struct Element {
        static void pow(const f32* __restrict Xx, f32 exp, f32* __restrict Xz, size_t N);
        static void sqrt(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void log(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void exp(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
    };
} //namespace cortex::_fw::stl

#endif //CORTEXMIND_CORE_ENGINE_STD_ELEMENT_WISE_HPP