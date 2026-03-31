//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_OPS_H
#define CORTEXMIND_CORE_TOOLS_OPS_H

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda::ops {
    struct Add {
        __device__ f32 operator()(const f32 a, const f32 b) const {
            return a + b;
        }
    };
    struct Sub {
        __device__ f32 operator()(const f32 a, const f32 b) const {
            return a - b;
        }
    };
    struct Mul {
        __device__ f32 operator()(const f32 a, const f32 b) const {
            return a * b;
        }
    };
    struct Div {
        __device__ f32 operator()(const f32 a, const f32 b) const {
            return a / b;
        }
    };
    struct ReLU {
        __device__ f32 operator()(const f32 a) const {
            return fmaxf(0.0f, a);
        }
    };
    struct LeakyReLU {
        f32 alpha;
        __device__ f32 operator()(const f32 a) const {
            return a > 0 ? a : a * this->alpha;
        }
    };
    struct Tanh {
        __device__ f32 operator()(const f32 a) const {
            return tanhf(a);
        }
    };
    struct GeLU {
        __device__ f32 operator()(const f32 a) const {
            constexpr f32 c0 = 0.7978845608028654f;
            constexpr f32 c1 = 0.044715f;
            const f32 inner  = c0 * (a + c1 * a * a * a);
            return 0.5f * a * (1.0f + tanhf(inner));
        }
    };
    struct SiLU {
        __device__ f32 operator()(const f32 a) const {
            return a / (1.0f + expf(-a));
        }
    };
    struct Swish {
        f32 beta;
        __device__ f32 operator()(const f32 a) const {
            return a / (1.0f + expf(-this->beta * a));
        }
    };
    struct Sigmoid {
        __device__ f32 operator()(const f32 a) const {
            return 1.0f / (1.0f + expf(-a));
        }
    };
} //namespace cortex::_fw::cuda::ops

#endif //CORTEXMIND_CORE_TOOLS_OPS_H