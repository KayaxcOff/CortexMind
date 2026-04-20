//
// Created by muham on 20.04.2026.
//

#include "CortexMind/tools/arithmetic_operations.hpp"

using namespace cortex;

tensor cortex::add(const tensor &Xx, const tensor &Xy) {
    return Xx + Xy;
}

tensor cortex::subtract(const tensor &Xx, const tensor &Xy) {
    return Xx - Xy;
}

tensor cortex::multiply(const tensor &Xx, const tensor &Xy) {
    return Xx * Xy;
}

tensor cortex::divide(const tensor &Xx, const tensor &Xy) {
    return Xx / Xy;
}