//
// Created by muham on 29.03.2026.
//

#include "CortexMind/core/Engine/AVX2/cmp.hpp"

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

vec8f cmp::gt(const vec8f Xx, const vec8f Xy) {
    return _mm256_cmp_ps(Xx, Xy, _CMP_GT_OQ);
}

vec8f cmp::lt(const vec8f Xx, const vec8f Xy) {
    return _mm256_cmp_ps(Xx, Xy, _CMP_LT_OQ);
}

vec8f cmp::eq(const vec8f Xx, const vec8f Xy) {
    return _mm256_cmp_ps(Xx, Xy, _CMP_EQ_OQ);
}

vec8f cmp::ge(const vec8f Xx, const vec8f Xy) {
    return _mm256_cmp_ps(Xx, Xy, _CMP_GE_OQ);
}

vec8f cmp::le(const vec8f Xx, const vec8f Xy) {
    return _mm256_cmp_ps(Xx, Xy, _CMP_LE_OQ);
}

vec8f cmp::ne(const vec8f Xx, const vec8f Xy) {
    return _mm256_cmp_ps(Xx, Xy, _CMP_NEQ_OQ);
}

i32 cmp::mask(const vec8f x) {
    return _mm256_movemask_ps(x);
}

bool cmp::any(const vec8f x) {
    return  _mm256_movemask_ps(x) != 0;
}

bool cmp::all(const vec8f x) {
    return _mm256_movemask_ps(x) == 0xFF;
}