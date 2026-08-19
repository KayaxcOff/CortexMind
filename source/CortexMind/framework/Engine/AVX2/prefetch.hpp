//
// Created by muham on 19.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_PREFETCH_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_PREFETCH_HPP

#include <immintrin.h>

namespace cortex::_fw::avx2 {
    struct prefetch {
        static void nta(const void* __restrict Xx) {
            _mm_prefetch(static_cast<const char*>(Xx), _MM_HINT_NTA);
        }
        static void t0(const void* __restrict Xx) {
            _mm_prefetch(static_cast<const char*>(Xx), _MM_HINT_T0);
        }
        static void t1(const void* __restrict Xx) {
            _mm_prefetch(static_cast<const char*>(Xx), _MM_HINT_T1);
        }
        static void t2(const void* __restrict Xx) {
            _mm_prefetch(static_cast<const char*>(Xx), _MM_HINT_T2);
        }
        static void enta(const void* __restrict Xx) {
            _mm_prefetch(static_cast<const char*>(Xx), _MM_HINT_ENTA);
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_PREFETCH_HPP