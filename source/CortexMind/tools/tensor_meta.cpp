//
// Created by muham on 20.05.2026.
//

#include "CortexMind/tools/tensor_meta.hpp"

using namespace cortex;

int32 cortex::argmax(const tensor &x) {
    const float32 max = x.max();

    int32 output = -1;
    for (size_t i = 0; i < x.len(); ++i) {
        if (x.get()[i] == max) {
            output = static_cast<int32>(i);
        }
    }
    return output;
}

int32 cortex::argmin(const tensor &x) {
    const float32 min = x.max();

    int32 output = -1;
    for (size_t i = 0; i < x.len(); ++i) {
        if (x.get()[i] == min) {
            output = static_cast<int32>(i);
        }
    }
    return output;
}
