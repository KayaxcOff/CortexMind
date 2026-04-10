//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_OPERATIONS_H
#define CORTEXMIND_CORE_TOOLS_OPERATIONS_H

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda::ops {
    struct Addition {
        __device__ f32 operator()(const f32 Xx, const f32 Xy) const {
            return Xx + Xy;
        }
    };
    struct Subtraction {
        __device__ f32 operator()(const f32 Xx, const f32 Xy) const {
            return Xx - Xy;
        }
    };
    struct Multiplication {
        __device__ f32 operator()(const f32 Xx, const f32 Xy) const {
            return Xx * Xy;
        }
    };
    struct Division {
        __device__ f32 operator()(const f32 Xx, const f32 Xy) const {
            return Xx / Xy;
        }
    };
    struct ReLU {
        __device__ f32 operator()(const f32 x) const {
            return fmaxf(0.0f, x);
        }
    };
    struct LeakyReLU {
        f32 alpha;
        explicit __host__ __device__ LeakyReLU(const f32 alpha = 0.01f) : alpha(alpha) {}
        __device__ f32 operator()(const f32 x) const {
            return x > 0.0f ? x : x * this->alpha;
        }
    };
    struct Sigmoid {
        __device__ f32 operator()(const f32 x) const {
            return 1.0f / (1.0f + expf(-x));
        }
    };
    struct SigmoidFast {
        __device__ f32 operator()(const f32 x) const {
            return __fdividef(1.0f, 1.0f + __expf(-x));
        }
    };
    struct Tanh {
        __device__ f32 operator()(const f32 x) const {
            return tanhf(x);
        }
    };
    struct GELU {
        __device__ f32 operator()(const f32 x) const {
            constexpr f32 c0 = 0.7978845608028654f;
            constexpr f32 c1 = 0.044715f;
            return 0.5f * x * (1.0f + tanhf(c0 * (x + c1 * x * x * x)));
        }
    };
    struct GELUExact {
        __device__ f32 operator()(const f32 x) const {
            return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
        }
    };
    struct SiLU {
        __device__ f32 operator()(const f32 x) const {
            return x / (1.0f + expf(-x));
        }
    };
    struct SiLUFast {
        __device__ f32 operator()(const f32 x) const {
            return __fdividef(x, 1.0f + __expf(-x));
        }
    };
    struct Swish {
        f32 beta;
        explicit __host__ __device__ Swish(const f32 beta = 1.0f) : beta(beta) {}
        __device__ f32 operator()(const f32 x) const {
            return x / (1.0f + expf(-this->beta * x));
        }
    };
    struct SwishFast {
        f32 beta;
        explicit __host__ __device__ SwishFast(const f32 beta = 1.0f) : beta(beta) {}
        __device__ f32 operator()(const f32 x) const {
            return __fdividef(x, 1.0f + __expf(-this->beta * x));
        }
    };
} //namespace cortex::_fw::cuda::ops

#endif //CORTEXMIND_CORE_TOOLS_OPERATIONS_H