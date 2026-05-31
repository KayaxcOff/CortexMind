//
// Created by muham on 20.05.2026.
//

#include "CortexMind/tools/tensor_meta.hpp"

using namespace cortex;

int32 cortex::argmax(const tensor &x) {
    const float32 max = x.max();

    for (size_t i = 0; i < x.len(); ++i) {
        if (x.get()[i] == max) {
            return static_cast<int32>(i);
        }
    }
    return -1;
}

int32 cortex::argmin(const tensor &x) {
    const float32 min = x.min();

    for (size_t i = 0; i < x.len(); ++i) {
        if (x.get()[i] == min) {
            return static_cast<int32>(i);
        }
    }
    return -1;
}
