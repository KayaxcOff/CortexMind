//
// Created by muham on 12.08.2026.
//

#include "CortexMind/framework/Engine/Operations/kernel_ops.cuh"

using namespace cortex::_fw::ops;

CXM_DEVICE
float Addition::operator()(const float Xx, const float Xy) const noexcept {
    return Xx + Xy;
}

CXM_DEVICE
tlx::bfloat16 Addition::operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
    return Xx + Xy;
}

CXM_DEVICE
tlx::half Addition::operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
    return Xx + Xy;
}

CXM_DEVICE
float Subtraction::operator()(const float Xx, const float Xy) const noexcept {
    return Xx - Xy;
}

CXM_DEVICE
tlx::bfloat16 Subtraction::operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
    return Xx - Xy;
}

CXM_DEVICE
tlx::half Subtraction::operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
    return Xx - Xy;
}

CXM_DEVICE
float Multiplication::operator()(const float Xx, const float Xy) const noexcept {
    return Xx * Xy;
}

CXM_DEVICE
tlx::bfloat16 Multiplication::operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
    return Xx * Xy;
}

CXM_DEVICE
tlx::half Multiplication::operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
    return Xx * Xy;
}

CXM_DEVICE
float Division::operator()(const float Xx, const float Xy) const noexcept {
    return Xx / Xy;
}

CXM_DEVICE
tlx::bfloat16 Division::operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
    return Xx / Xy;
}

CXM_DEVICE
tlx::half Division::operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
    return Xx / Xy;
}

CXM_DEVICE
float Square::operator()(const float Xx) const noexcept {
    return Xx * Xx;
}

CXM_DEVICE
tlx::bfloat16 Square::operator()(const tlx::bfloat16 Xx) const noexcept {
    return Xx * Xx;
}

CXM_DEVICE
tlx::half Square::operator()(const tlx::half Xx) const noexcept {
    return Xx * Xx;
}

CXM_DEVICE
float Power::operator()(const float Xx, const float Xy) const noexcept {
    return powf(Xx, Xy);
}

CXM_DEVICE
tlx::bfloat16 Power::operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
    return powf(Xx, Xy);
}

CXM_DEVICE
tlx::half Power::operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
    return powf(Xx, Xy);
}

CXM_DEVICE
float Sqrt::operator()(const float Xx) const noexcept {
    return sqrtf(Xx);
}

CXM_DEVICE
tlx::bfloat16 Sqrt::operator()(const tlx::bfloat16 Xx) const noexcept {
    return hsqrt(Xx);
}

CXM_DEVICE
tlx::half Sqrt::operator()(const tlx::half Xx) const noexcept {
    return hsqrt(Xx);
}

CXM_DEVICE
float RSqrt::operator()(const float Xx) const noexcept {
    return rsqrtf(Xx);
}

CXM_DEVICE
tlx::bfloat16 RSqrt::operator()(const tlx::bfloat16 Xx) const noexcept {
    return hrsqrt(Xx);
}

CXM_DEVICE
tlx::half RSqrt::operator()(const tlx::half Xx) const noexcept {
    return hrsqrt(Xx);
}

CXM_DEVICE
float Log::operator()(const float Xx) const noexcept {
    return logf(Xx);
}

CXM_DEVICE
tlx::bfloat16 Log::operator()(const tlx::bfloat16 Xx) const noexcept {
    return hlog(Xx);
}

CXM_DEVICE
tlx::half Log::operator()(const tlx::half Xx) const noexcept {
    return hlog(Xx);
}

CXM_DEVICE
float Exp::operator()(const float Xx) const noexcept {
    return expf(Xx);
}

CXM_DEVICE
tlx::bfloat16 Exp::operator()(const tlx::bfloat16 Xx) const noexcept {
    return hexp(Xx);
}

CXM_DEVICE
tlx::half Exp::operator()(const tlx::half Xx) const noexcept {
    return hexp(Xx);
}

CXM_DEVICE
float Erf::operator()(const float Xx) const noexcept {
    return erff(Xx);
}

CXM_DEVICE
tlx::bfloat16 Erf::operator()(const tlx::bfloat16 Xx) const noexcept {
    return erff(Xx);
}

CXM_DEVICE
tlx::half Erf::operator()(const tlx::half Xx) const noexcept {
    return erff(Xx);
}

CXM_DEVICE
float Sin::operator()(const float Xx) const noexcept {
    return sinf(Xx);
}

CXM_DEVICE
tlx::bfloat16 Sin::operator()(const tlx::bfloat16 Xx) const noexcept {
    return hsin(Xx);
}

CXM_DEVICE
tlx::half Sin::operator()(const tlx::half Xx) const noexcept {
    return hsin(Xx);
}

CXM_DEVICE
float Abs::operator()(const float Xx) const noexcept {
    return fabsf(Xx);
}

CXM_DEVICE
tlx::bfloat16 Abs::operator()(const tlx::bfloat16 Xx) const noexcept {
    return __habs(Xx);
}

CXM_DEVICE
tlx::half Abs::operator()(const tlx::half Xx) const noexcept {
    return __habs(Xx);
}

CXM_DEVICE
float Neg::operator()(const float Xx) const noexcept {
    return -Xx;
}

CXM_DEVICE
tlx::bfloat16 Neg::operator()(const tlx::bfloat16 Xx) const noexcept {
    return __hneg(Xx);
}

CXM_DEVICE
tlx::half Neg::operator()(const tlx::half Xx) const noexcept {
    return __hneg(Xx);
}

CXM_DEVICE
float Rcp::operator()(const float Xx) const noexcept {
    return 1.f / Xx;
}

CXM_DEVICE
tlx::bfloat16 Rcp::operator()(const tlx::bfloat16 Xx) const noexcept {
    return hrcp(Xx);
}

CXM_DEVICE
tlx::half Rcp::operator()(const tlx::half Xx) const noexcept {
    return hrcp(Xx);
}

CXM_DEVICE
float Lerp::operator()(const float Xx, const float Xy, const float Xz) const noexcept {
    return fmaf(Xz, Xy - Xx, Xx);
}

CXM_DEVICE
tlx::bfloat16 Lerp::operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy, const tlx::bfloat16 Xz) const noexcept {
    return __hfma(Xz, Xy - Xx, Xy);
}

CXM_DEVICE
tlx::half Lerp::operator()(const tlx::half Xx, const tlx::half Xy, const tlx::half Xz) const noexcept {
    return __hfma(Xz, Xy - Xx, Xx);
}