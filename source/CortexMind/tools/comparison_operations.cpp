//
// Created by muham on 20.04.2026.
//

#include "CortexMind/tools/comparison_operations.hpp"
#include <CortexMind/core/Engine/AVX2/cmp.hpp>
#include <CortexMind/core/Engine/AVX2/functions.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/core/Engine/CUDA/comparison.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Memory/device.hpp>
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::sys;
using namespace cortex::_fw;
using namespace cortex;

int32 cortex::argmax(const tensor &x) {
    const float32 max = x.max();

    for (int32 i = 0; i < static_cast<int32>(x.numel()); ++i) {
        if (max == x.get()[i]) {
            return i;
        }
    }
    return -1;
}

int32 cortex::argmin(const tensor &x) {
    const float32 min = x.min();

    for (int32 i = 0; i < static_cast<int32>(x.numel()); ++i) {
        if (min == x.get()[i]) {
            return i;
        }
    }
    return -1;
}

boolean cortex::greater(const tensor &Xx, const tensor &Xy) {
    return false;
}

boolean cortex::less(const tensor &Xx, const tensor &Xy) {
    return false;
}

boolean cortex::equal(const tensor &Xx, const tensor &Xy) {
    return false;
}