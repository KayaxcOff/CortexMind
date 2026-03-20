//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_KERNEL_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_KERNEL_CUH

#include <CortexMind/core/Engine/CUDA/params.cuh>
#include <CortexMind/core/Engine/CUDA/utils.cuh>

// MatMul and FMA require more variables as input
// values compared to other functions. Therefore,
// to avoid compatibility issues, they were written
// as separate kernels.

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief   Generic element-wise binary operation: Z[i] = op(X[i], Y[i])
     * @tparam  Op      Functor type (e.g. op::Add, op::Mul, op::Div)
     * @param   Xx      First input array (const float4 view)
     * @param   Xy      Second input array (const float4 view)
     * @param   Xz      Output array (float4 view)
     * @param   idx     Total number of float elements (not float4 count)
     *
     * @note    Main loop processes 4 elements per iteration
     * @note    Tail (remaining 0–3 elements) handled via scalar path
     * @note    Op must be `__device__` callable with signature f32(f32, f32)
     * @note    All pointers must be 16-byte aligned
     */
    template<typename Op>
    __global__ void matrix_kernel(const f4x32* __restrict Xx, const f4x32* __restrict Xy, f4x32* __restrict Xz, size_t idx) {
        Op op;
        CXM_CUDA_LOOP(i, idx / 4) {
            Xz[i] = {op(Xx[i].x, Xy[i].x), op(Xx[i].y, Xy[i].y),
                     op(Xx[i].z, Xy[i].z), op(Xx[i].w, Xy[i].w)};
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid  = global_thread_id();
        if (tid < tail) {
            reinterpret_cast<f32*>(Xz)[base + tid] =
                op(reinterpret_cast<const f32*>(Xx)[base + tid],
                   reinterpret_cast<const f32*>(Xy)[base + tid]);
        }
    }
    /**
     * @brief   Element-wise fused multiply-add: Z[i] = X[i] × Y[i] + W[i]
     * @param   Xx      Multiplicand array (const float4 view)
     * @param   Xy      Multiplier array (const float4 view)
     * @param   Xz      Addend array (const float4 view)
     * @param   Xk      Output array (float4 view)
     * @param   idx     Total number of float elements
     *
     * @note    Uses __fmaf_rn intrinsic (round-to-nearest) for better numerical stability
     * @note    Vectorized main loop + scalar tail handling
     * @note    All pointers must be 16-byte aligned
     */
    __global__ void matrix_fma_kernel(const f4x32* __restrict Xx, const f4x32* __restrict Xy, const f4x32* __restrict Xz, f4x32* __restrict Xk, size_t idx) {
        CXM_CUDA_LOOP(i, idx / 4) {
            Xk[i] = {__fmaf_rn(Xx[i].x, Xy[i].x, Xz[i].x),
                     __fmaf_rn(Xx[i].y, Xy[i].y, Xz[i].y),
                     __fmaf_rn(Xx[i].z, Xy[i].z, Xz[i].z),
                     __fmaf_rn(Xx[i].w, Xy[i].w, Xz[i].w)};
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid  = global_thread_id();
        if (tid < tail) {
            reinterpret_cast<f32*>(Xk)[base + tid] =
                __fmaf_rn(reinterpret_cast<const f32*>(Xx)[base + tid],
                          reinterpret_cast<const f32*>(Xy)[base + tid],
                          reinterpret_cast<const f32*>(Xz)[base + tid]);
        }
    }
    /**
     * @brief   Dense matrix multiplication: Z = X × Y   (row-major, tiled shared memory)
     * @param   X       Matrix A (M × K), row-major
     * @param   Y       Matrix B (K × N), row-major
     * @param   Z       Output matrix C (M × N), row-major
     * @param   M       Rows in X and Z
     * @param   K       Columns in X = rows in Y
     * @param   N       Columns in Y and Z
     *
     * @note    Uses classic tiled shared memory approach (MAT_TILE × MAT_TILE)
     * @note    Each thread computes one output element (or zero if out-of-bounds)
     * @note    Boundary handling via zero-padding in shared memory
     * @note    Uses __fmaf_rn for accumulation (good numerical stability)
     * @note    Launch recommendation:
     *          dim3 block(MAT_TILE, MAT_TILE);
     *          dim3 grid = grid2d(M, N, MAT_TILE);
     *          matrix_matmul_kernel<<<grid, block>>>(...)
     * @note    Basic implementation — suitable for small-to-medium matrices
     *          For large matrices consider cuBLAS, CUTLASS or WMMA
     */
    __global__ void matrix_matmul_kernel(const f32* __restrict X, const f32* __restrict Y, f32* __restrict Z, size_t M, size_t K, size_t N) {
        __shared__ f32 tileX[MAT_TILE][MAT_TILE];
        __shared__ f32 tileY[MAT_TILE][MAT_TILE];

        const size_t row = blockIdx.y * MAT_TILE + threadIdx.y;
        const size_t col = blockIdx.x * MAT_TILE + threadIdx.x;

        f32 acc = 0.0f;

        for (size_t t = 0; t < (K + MAT_TILE - 1) / MAT_TILE; ++t) {
            const size_t kx = t * MAT_TILE + threadIdx.x;
            const size_t ky = t * MAT_TILE + threadIdx.y;

            tileX[threadIdx.y][threadIdx.x] = (row < M && kx < K) ? X[row * K + kx] : 0.0f;
            tileY[threadIdx.y][threadIdx.x] = (ky < K && col < N) ? Y[ky * N + col] : 0.0f;
            __syncthreads();

            for (size_t k = 0; k < MAT_TILE; ++k)
                acc = __fmaf_rn(tileX[threadIdx.y][k], tileY[k][threadIdx.x], acc);
            __syncthreads();
        }

        if (row < M && col < N) Z[row * N + col] = acc;
    }

    __global__ void matrix_fill_kernel(f4x32* __restrict__ Xx, f32 value, size_t idx) {
        const f4x32 val4 = {value, value, value, value};
        CXM_CUDA_LOOP(i, idx / 4) Xx[i] = val4;
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid  = global_thread_id();
        if (tid < tail) reinterpret_cast<f32*>(Xx)[base + tid] = value;
    }

    __global__ void matrix_sqrt_kernel(const f4x32* __restrict__ Xx, f4x32* __restrict__ Xz, size_t idx) {
        CXM_CUDA_LOOP(i, idx / 4) {
            Xz[i] = {sqrtf(Xx[i].x), sqrtf(Xx[i].y),
                     sqrtf(Xx[i].z), sqrtf(Xx[i].w)};
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid  = global_thread_id();
        if (tid < tail)
            reinterpret_cast<f32*>(Xz)[base + tid] =
                sqrtf(reinterpret_cast<const f32*>(Xx)[base + tid]);
    }

    __global__ void matrix_pow_kernel(const f4x32* __restrict__ Xx, f32 value, f4x32* __restrict__ Xz, size_t idx) {
        CXM_CUDA_LOOP(i, idx / 4) {
            Xz[i] = {powf(Xx[i].x, value), powf(Xx[i].y, value),
                     powf(Xx[i].z, value), powf(Xx[i].w, value)};
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid  = global_thread_id();
        if (tid < tail)
            reinterpret_cast<f32*>(Xz)[base + tid] =
                powf(reinterpret_cast<const f32*>(Xx)[base + tid], value);
    }
} // namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_KERNEL_CUH