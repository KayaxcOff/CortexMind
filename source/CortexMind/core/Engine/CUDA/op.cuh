//
// Created by muham on 16.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_OP_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_OP_CUH

#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::cuda::op {
    /**
     * @struct  Add
     * @brief   Element-wise addition: x + val
     */
    struct Add {
        __device__ inline f32 operator()(f32 x, f32 val) const {
            return x + val;
        }
    };
    /**
     * @struct  Sub
     * @brief   Element-wise subtraction: x - val
     */
    struct Sub {
        __device__ inline f32 operator()(f32 x, f32 val) const {
            return x - val;
        }
    };
    /**
     * @struct  Mul
     * @brief   Element-wise multiplication: x × val
     */
    struct Mul {
        __device__ inline f32 operator()(f32 x, f32 val) const {
            return x * val;
        }
    };
    /**
     * @struct  Div
     * @brief   Element-wise division: x ÷ val  (using reciprocal multiply)
     * @warning val == 0 → Inf or NaN
     */
    struct Div {
        __device__ inline f32 operator()(f32 x, f32 val) const {
            return x * (1.0f / val);
        }
    };
    /**
     * @struct  Relu
     * @brief   Rectified Linear Unit: max(x, 0)
     */
    struct Relu {
        __device__ inline f32 operator()(f32 x) const {
            return x > 0.0f ? x : 0.0f;
        }
    };
    /**
     * @struct  Sigmoid
     * @brief   Sigmoid activation: 1 / (1 + exp(-x))
     */
    struct Sigmoid {
        __device__ inline f32 operator()(f32 x) const {
            return 1.0f / (1.0f + expf(-x));
        }
    };
    /**
     * @struct  SigmoidFast
     * @brief   Fast sigmoid approximation using reciprocal intrinsic
     */
    struct SigmoidFast {
        __device__ inline f32 operator()(f32 x) const {
            return __frcp_rn(1.0f + expf(-x));
        }
    };
    /**
     * @struct  Gelu
     * @brief   Gaussian Error Linear Unit (approximate): 0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))
     */
    struct Gelu {
        __device__ inline f32 operator()(f32 x) const {
            return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        }
    };
    /**
     * @struct  GeluExact
     * @brief   Exact GELU using erf: 0.5 x (1 + erf(x / √2))
     * @note    Requires erff support (may need -use_fast_math or SVML)
     */
    struct GeluExact {
        __device__ inline f32 operator()(f32 x) const {
            return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
        }
    };
    /**
     * @struct  Silu
     * @brief   Sigmoid Linear Unit (Swish with β=1): x × sigmoid(x)
     */
    struct Silu {
        __device__ inline f32 operator()(f32 x) const {
            return x * (1.0f / (1.0f + expf(-x)));
        }
    };
    /**
     * @struct  Exp
     * @brief   Element-wise exponential: exp(x)
     */
    struct Exp {
        __device__ inline f32 operator()(f32 x) const {
            return expf(x);
        }
    };
    /**
     * @struct  Log
     * @brief   Element-wise natural logarithm: log(x)
     * @note    x ≤ 0 produces NaN or -Inf
     */
    struct Log {
        __device__ inline f32 operator()(f32 x) const {
            return logf(x);
        }
    };
    /**
     * @struct  Abs
     * @brief   Element-wise absolute value: |x|
     */
    struct Abs {
        __device__ inline f32 operator()(f32 x) const {
            return fabsf(x);
        }
    };
} // namespace cortex::_fw::cuda::op

#endif //CORTEXMIND_CORE_ENGINE_CUDA_OP_CUH