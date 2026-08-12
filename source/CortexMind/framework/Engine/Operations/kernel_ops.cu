//
// Created by muham on 11.08.2026.
//

#include "CortexMind/framework/Engine/Operations/kernel_ops.cuh"

using namespace cortex::_fw::ops;

CXM_DEVICE_ATTR float Addition::operator()(const float Xx, const float Xy) const noexcept {
    return Xx + Xy;
}

CXM_DEVICE_ATTR tlx::bfloat16 Addition::operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
    return Xx + Xy;
}

CXM_DEVICE_ATTR tlx::half Addition::operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
    return Xx + Xy;
}

CXM_DEVICE_ATTR float Subtraction::operator()(const float Xx, const float Xy) const noexcept {
    return Xx - Xy;
}

CXM_DEVICE_ATTR tlx::bfloat16 Subtraction::operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
    return Xx - Xy;
}

CXM_DEVICE_ATTR tlx::half Subtraction::operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
    return Xx - Xy;
}

CXM_DEVICE_ATTR float Multiplication::operator()(const float Xx, const float Xy) const noexcept {
    return Xx * Xy;
}

CXM_DEVICE_ATTR tlx::bfloat16 Multiplication::operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
    return Xx * Xy;
}

CXM_DEVICE_ATTR tlx::half Multiplication::operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
    return Xx * Xy;
}

CXM_DEVICE_ATTR float Square::operator()(const float Xx) const noexcept {
    return Xx * Xx;
}

CXM_DEVICE_ATTR tlx::bfloat16 Square::operator()(const tlx::bfloat16 Xx) const noexcept {
    return Xx * Xx;
}

CXM_DEVICE_ATTR tlx::half Square::operator()(const tlx::half Xx) const noexcept {
    return Xx * Xx;
}

CXM_DEVICE_ATTR float Power::operator()(const float Xx, const float Xy) const noexcept {
    return powf(Xx, Xy);
}

CXM_DEVICE_ATTR tlx::bfloat16 Power::operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
    return powf(Xx, Xy);
}

CXM_DEVICE_ATTR tlx::half Power::operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
    return powf(Xx, Xy);
}

CXM_DEVICE_ATTR float Sqrt::operator()(const float Xx) const noexcept {
    return sqrtf(Xx);
}

CXM_DEVICE_ATTR tlx::bfloat16 Sqrt::operator()(const tlx::bfloat16 Xx) const noexcept {
    return sqrtf(Xx);
}

CXM_DEVICE_ATTR tlx::half Sqrt::operator()(const tlx::half Xx) const noexcept {
    return sqrtf(Xx);
}

CXM_DEVICE_ATTR float RSqrt::operator()(const float Xx) const noexcept {
    return rsqrtf(Xx);
}

CXM_DEVICE_ATTR tlx::bfloat16 RSqrt::operator()(const tlx::bfloat16 Xx) const noexcept {
    return rsqrtf(Xx);
}

CXM_DEVICE_ATTR tlx::half RSqrt::operator()(const tlx::half Xx) const noexcept {
    return rsqrtf(Xx);
}

CXM_DEVICE_ATTR float Log::operator()(const float Xx) const noexcept {
    return logf(Xx);
}

CXM_DEVICE_ATTR tlx::bfloat16 Log::operator()(const tlx::bfloat16 Xx) const noexcept {
    return logf(Xx);
}

CXM_DEVICE_ATTR tlx::half Log::operator()(const tlx::half Xx) const noexcept {
    return logf(Xx);
}

CXM_DEVICE_ATTR float Exp::operator()(const float Xx) const noexcept {
    return expf(Xx);
}

CXM_DEVICE_ATTR tlx::bfloat16 Exp::operator()(const tlx::bfloat16 Xx) const noexcept {
    return expf(Xx);
}

CXM_DEVICE_ATTR tlx::half Exp::operator()(const tlx::half Xx) const noexcept {
    return expf(Xx);
}

#ifdef __CUDACC_RELAXED_CONSTEXPR__
#pragma message("TLX: CUDA relaxed constexpr ENABLED")
#else
#pragma message("TLX: CUDA relaxed constexpr DISABLED")
#endif