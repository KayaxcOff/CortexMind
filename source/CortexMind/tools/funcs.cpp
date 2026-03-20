//
// Created by muham on 20.03.2026.
//

#include "CortexMind/tools/funcs.hpp"

using namespace cortex;

tensor cortex::addition(tensor &Xx, const tensor &Yy) noexcept {
    return Xx + Yy;
}

tensor cortex::subtract(tensor &Xx, const tensor &Yy) noexcept {
    return Xx - Yy;
}

tensor cortex::multiply(tensor &Xx, const tensor &Yy) noexcept {
    return Xx * Yy;
}

tensor cortex::divide(tensor &Xx, const tensor &Yy) noexcept {
    return Xx / Yy;
}

tensor cortex::max(const tensor &Xx, const tensor &Yy) noexcept {
    return {};
}

tensor cortex::min(const tensor &Xx, const tensor &Yy) noexcept {
    return {};
}
