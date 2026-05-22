//
// Created by muham on 22.05.2026.
//

#include "CortexMind/tools/math.hpp"

using namespace cortex;

tensor cortex::add(const tensor &Xx, const tensor &Xy) {
    return Xx + Xy;
}

tensor cortex::sub(const tensor &Xx, const tensor &Xy) {
    return Xx - Xy;
}

tensor cortex::mul(const tensor &Xx, const tensor &Xy) {
    return Xx * Xy;
}

tensor cortex::div(const tensor &Xx, const tensor &Xy) {
    return Xx / Xy;
}

tensor cortex::matmul(const tensor &Xx, const tensor &Xy) {
    return Xx.matmul(Xy);
}