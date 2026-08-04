//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_BIT_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_BIT_HPP

#include <tlx/types.hpp>

namespace cortex::_fw {
    namespace avx2 {
        struct vec8f;
    } //namespace avx2

    namespace detail {
        void load_bf16(const tlx::bfloat16* src, avx2::vec8f& low, avx2::vec8f& high);
        void store_bf16(tlx::bfloat16* dst, const avx2::vec8f& low, const avx2::vec8f& high);
    } //namespace detail
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_BIT_HPP