//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_KERNEL_OPERATIONS_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_KERNEL_OPERATIONS_HPP

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::ops {
    /**
     * @brief Functor for addition operation.
     *
     * Used in generic CUDA kernels for element-wise addition.
     */
    struct Addition {
        __device__ __forceinline__ f32 operator()(const f32 Xx, const f32 Xy) const {
            return Xx + Xy;
        }
    };
    /**
     * @brief Functor for subtraction operation.
     *
     * Used in generic CUDA kernels for element-wise subtraction.
     */
    struct Subtraction {
        __device__ __forceinline__ f32 operator()(const f32 Xx, const f32 Xy) const {
            return Xx - Xy;
        }
    };
    /**
     * @brief Functor for multiplication operation.
     *
     * Used in generic CUDA kernels for element-wise multiplication.
     */
    struct Multiplication {
        __device__ __forceinline__ f32 operator()(const f32 Xx, const f32 Xy) const {
            return Xx * Xy;
        }
    };
    /**
     * @brief Functor for division operation.
     *
     * Used in generic CUDA kernels for element-wise division.
     */
    struct Division {
        __device__ __forceinline__ f32 operator()(const f32 Xx, const f32 Xy) const {
            return Xx / Xy;
        }
    };

    /**
     * @brief Functor for power operation (unary with parameter).
     */
    struct Pow {
        explicit __host__ __device__ Pow(const f32 e) : exp(e) {}

        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return __powf(x, this->exp);
        }

    private:
        f32 exp;
    };

    /**
     * @brief Functor for absolute value operation.
     */
    struct Abs {
        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return fabsf(x);
        }
    };

    /**
     * @brief Functor for square root operation.
     */
    struct Sqrt {
        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return __fsqrt_rn(x);
        }
    };

    /**
     * @brief Functor for squaring operation (x²).
     */
    struct Square {
        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return x * x;
        }
    };

    /**
     * @brief Functor for reciprocal square root (1/sqrt(x)).
     */
    struct Rsqrt {
        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return rsqrtf(x);
        }
    };

    /**
     * @brief Functor for sine operation.
     */
    struct Sin {
        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return sinf(x);
        }
    };

    /**
     * @brief Fast sine approximation using hardware intrinsics.
     */
    struct SinFast {
        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return __sinf(x);
        }
    };

    /**
     * @brief Functor for cosine operation.
     */
    struct Cos {
        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return cosf(x);
        }
    };

    /**
     * @brief Fast cosine approximation using hardware intrinsics.
     */
    struct CosFast {
        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return __cosf(x);
        }
    };

    /**
     * @brief Functor for natural logarithm.
     */
    struct Log {
        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return __logf(x);
        }
    };

    /**
     * @brief Functor for sign operation (Returns -1.0 for negative, 1.0 for positive, 0.0 for zero).
     */
    struct Sign {
        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return (0.0f < x) - (x < 0.0f);
        }
    };

    /**
     * @brief Functor for negation operation (-x).
     */
    struct Neg {
        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return -x;
        }
    };


    /**
     * @brief Functor for exponential (e^x).
     */
    struct Exp {
        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return __expf(x);
        }
    };

    /**
     * @brief ReLU activation functor: max(0, x)
     */
    struct ReLU {
        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return fmaxf(0.0f, x);
        }
    };

    /**
     * @brief Leaky ReLU activation functor.
     */
    struct LeakyReLU {
        explicit __host__ __device__ LeakyReLU(const f32 alpha = 0.01f) : alpha(alpha) {}

        __device__ __forceinline__ f32 operator()(const f32 x) const {
            return x > 0.0f ? x : x * this->alpha;
        }

    private:
        f32 alpha;
    };

    /**
     * @brief Sigmoid activation functor.
     */
    struct Sigmoid {
        __device__ f32 operator()(const f32 x) const {
            return 1.0f / (1.0f + expf(-x));
        }
    };

    /**
     * @brief Fast Sigmoid approximation using fast division.
     */
    struct SigmoidFast {
        __device__ f32 operator()(const f32 x) const {
            return __fdividef(1.0f, 1.0f + __expf(-x));
        }
    };

    /**
     * @brief Hyperbolic tangent activation functor.
     */
    struct Tanh {
        __device__ f32 operator()(const f32 x) const {
            return tanhf(x);
        }
    };

    /**
     * @brief GELU activation (tanh approximation).
     */
    struct GELU {
        __device__ f32 operator()(const f32 x) const {
            constexpr f32 c0 = 0.7978845608028654f;
            constexpr f32 c1 = 0.044715f;
            return 0.5f * x * (1.0f + tanhf(c0 * (x + c1 * x * x * x)));
        }
    };

    /**
     * @brief GELU activation using erf function (more accurate).
     */
    struct GELUExact {
        __device__ f32 operator()(const f32 x) const {
            return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
        }
    };

    /**
     * @brief SiLU (Sigmoid Linear Unit) activation.
     */
    struct SiLU {
        __device__ f32 operator()(const f32 x) const {
            return x / (1.0f + expf(-x));
        }
    };

    /**
     * @brief Fast SiLU approximation.
     */
    struct SiLUFast {
        __device__ f32 operator()(const f32 x) const {
            return __fdividef(x, 1.0f + __expf(-x));
        }
    };

    /**
     * @brief Swish activation with configurable beta.
     */
    struct Swish {
        explicit __host__ __device__ Swish(const f32 beta = 1.0f) : beta(beta) {}

        __device__ f32 operator()(const f32 x) const {
            return x / (1.0f + expf(-this->beta * x));
        }

    private:
        f32 beta;
    };

    /**
     * @brief Fast Swish approximation with configurable beta.
     */
    struct SwishFast {
        explicit __host__ __device__ SwishFast(const f32 beta = 1.0f) : beta(beta) {}

        __device__ f32 operator()(const f32 x) const {
            return __fdividef(x, 1.0f + __expf(-this->beta * x));
        }

    private:
        f32 beta;
    };

    struct Constant {
        explicit __host__ __device__ Constant(const f32 val) : value(val) {}
        __device__ __forceinline__ f32 operator()(const f32) const {
            return value;
        }
    private:
        f32 value;
    };
} //namespace cortex::_fw::ops

#endif //CORTEXMIND_FRAMEWORK_TOOLS_KERNEL_OPERATIONS_HPP