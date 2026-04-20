//
// Created by muham on 20.04.2026.
//

#include "CortexMind/tools/comparison_operations.hpp"

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