//
// Created by muham on 13.03.2026.
//

#include "CortexMind/core/Engine/AVX2/cmp.hpp"

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

vec8f cmp::gt(const vec8f &x, const vec8f &y) {
    return _mm256_cmp_ps(x, y, _CMP_GT_OS);
}

vec8f cmp::lt(const vec8f &x, const vec8f &y) {
    return _mm256_cmp_ps(x, y, _CMP_LT_OS);
}

vec8f cmp::eq(const vec8f &x, const vec8f &y) {
    return _mm256_cmp_ps(x, y, _CMP_EQ_OS);
}

vec8f cmp::ge(const vec8f &x, const vec8f &y) {
    return _mm256_cmp_ps(x, y, _CMP_GE_OS);
}

vec8f cmp::le(const vec8f &x, const vec8f &y) {
    return _mm256_cmp_ps(x, y, _CMP_LE_OS);
}

vec8f cmp::ne(const vec8f &x, const vec8f &y) {
    return _mm256_cmp_ps(x, y, _CMP_NEQ_OS);
}

i32 cmp::mask(const vec8f &x) {
    return _mm256_movemask_ps(x);
}

bool cmp::any(const vec8f &x) {
    return _mm256_movemask_ps(x) != 0;
}

bool cmp::all(const vec8f &x) {
    return _mm256_movemask_ps(x) == 0xFF;
}