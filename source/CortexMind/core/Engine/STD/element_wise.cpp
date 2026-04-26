//
// Created by muham on 26.04.2026.
//

#include "CortexMind/core/Engine/STD/element_wise.hpp"
#include <cmath>

using namespace cortex::_fw::stl;

void Element::pow(const f32 *Xx, const f32 exp, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = std::pow(Xx[i], exp);
    }
}

void Element::sqrt(const f32 *Xx, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = std::sqrt(Xx[i]);
    }
}

void Element::log(const f32 *Xx, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = std::log(Xx[i]);
    }
}

void Element::exp(const f32 *Xx, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = std::exp(Xx[i]);
    }
}