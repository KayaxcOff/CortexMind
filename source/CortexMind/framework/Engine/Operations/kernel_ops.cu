//
// Created by muham on 11.08.2026.
//

#include "CortexMind/framework/Engine/Operations/kernel_ops.cuh"

using namespace cortex::_fw::ops;

float Addition::operator()(const float Xx, const float Xy) const noexcept {
    return Xx + Xy;
}

tlx::bfloat16 Addition::operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
    return Xx + Xy;
}

tlx::half Addition::operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
    return Xx + Xy;
}