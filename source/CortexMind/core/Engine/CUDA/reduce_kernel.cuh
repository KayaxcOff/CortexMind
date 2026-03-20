//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_KERNEL_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_KERNEL_CUH

#include <CortexMind/core/Engine/CUDA/params.cuh>
#include <CortexMind/core/Engine/CUDA/utils.cuh>
#include <math_constants.h>

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief   Warp-level sum reduction using shuffle instructions
     * @param   val     Per-lane input value
     * @return  Sum of all values in the warp (broadcast to all lanes)
     *
     * @note    Uses __shfl_down_sync with full mask (0xffffffff)
     * @note    Assumes warp size = 32 (WARP_SIZE)
     */
    __device__ inline f32 warp_reduce_sum(f32 val) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        return val;
    }
    /**
     * @brief   Warp-level max reduction
     * @param   val     Per-lane input value
     * @return  Maximum value in the warp (broadcast to all lanes)
     */
    __device__ inline f32 warp_reduce_max(f32 val) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        return val;
    }
    /**
     * @brief   Warp-level min reduction
     * @param   val     Per-lane input value
     * @return  Minimum value in the warp (broadcast to all lanes)
     */
    __device__ inline f32 warp_reduce_min(f32 val) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
        return val;
    }
    /**
     * @brief   Computes horizontal sum of array: out = sum(X[0..idx-1])
     * @param   Xx      Input array (float4 view)
     * @param   out     Single-element output (device pointer)
     * @param   idx     Number of float elements
     *
     * @note    Vectorized loading (4 floats per iteration)
     * @note    Warp shuffle + shared memory + atomicAdd
     * @note    Final result written via atomicAdd (safe for multiple blocks)
     */
    __global__ void reduce_hsum_kernel(const f4x32* __restrict__ Xx, f32* __restrict__ out, size_t idx) {
        __shared__ f32 shared[WARP_SIZE];

        const size_t tid    = threadIdx.x;
        const int warp_id   = tid / WARP_SIZE;
        const int lane_id   = tid % WARP_SIZE;

        f32 val = 0.0f;
        CXM_CUDA_LOOP(i, idx / 4) {
            val += Xx[i].x + Xx[i].y + Xx[i].z + Xx[i].w;
        }
        // tail
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid2 = global_thread_id();
        if (tid2 < tail) val += reinterpret_cast<const f32*>(Xx)[base + tid2];

        val = warp_reduce_sum(val);
        if (lane_id == 0) shared[warp_id] = val;
        __syncthreads();

        val = (tid < blockDim.x / WARP_SIZE) ? shared[lane_id] : 0.0f;
        if (warp_id == 0) val = warp_reduce_sum(val);
        if (tid == 0) atomicAdd(out, val);
    }
    /**
     * @brief   Computes horizontal maximum: out = max(X[0..idx-1])
     * @param   Xx      Input array
     * @param   out     Single-element output
     * @param   idx     Number of elements
     *
     * @note    Initializes with -Inf
     * @note    Uses fmaxf for comparisons
     */
    __global__ void reduce_hmax_kernel(const f4x32* __restrict__ Xx, f32* __restrict__ out, size_t idx) {
        __shared__ f32 shared[WARP_SIZE];

        const size_t tid    = threadIdx.x;
        const int warp_id   = tid / WARP_SIZE;
        const int lane_id   = tid % WARP_SIZE;

        f32 val = -CUDART_INF_F;
        CXM_CUDA_LOOP(i, idx / 4) {
            val = fmaxf(val, fmaxf(fmaxf(Xx[i].x, Xx[i].y), fmaxf(Xx[i].z, Xx[i].w)));
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid2 = global_thread_id();
        if (tid2 < tail) val = fmaxf(val, reinterpret_cast<const f32*>(Xx)[base + tid2]);

        val = warp_reduce_max(val);
        if (lane_id == 0) shared[warp_id] = val;
        __syncthreads();

        val = (tid < blockDim.x / WARP_SIZE) ? shared[lane_id] : -CUDART_INF_F;
        if (warp_id == 0) val = warp_reduce_max(val);
        if (tid == 0) out[0] = val;
    }
    /**
     * @brief   Computes horizontal minimum: out = min(X[0..idx-1])
     * @param   Xx      Input array
     * @param   out     Single-element output
     * @param   idx     Number of elements
     *
     * @note    Initializes with +Inf
     * @note    Uses fminf for comparisons
     */
    __global__ void reduce_hmin_kernel(const f4x32* __restrict__ Xx, f32* __restrict__ out, size_t idx) {
        __shared__ f32 shared[WARP_SIZE];

        const size_t tid    = threadIdx.x;
        const int warp_id   = tid / WARP_SIZE;
        const int lane_id   = tid % WARP_SIZE;

        f32 val = CUDART_INF_F;
        CXM_CUDA_LOOP(i, idx / 4) {
            val = fminf(val, fminf(fminf(Xx[i].x, Xx[i].y), fminf(Xx[i].z, Xx[i].w)));
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid2 = global_thread_id();
        if (tid2 < tail) val = fminf(val, reinterpret_cast<const f32*>(Xx)[base + tid2]);

        val = warp_reduce_min(val);
        if (lane_id == 0) shared[warp_id] = val;
        __syncthreads();

        val = (tid < blockDim.x / WARP_SIZE) ? shared[lane_id] : CUDART_INF_F;
        if (warp_id == 0) val = warp_reduce_min(val);
        if (tid == 0) out[0] = val;
    }
    /**
     * @brief   Computes mean of array: out = sum(X) / idx
     * @param   Xx      Input array
     * @param   out     Single-element output
     * @param   idx     Number of elements (must be > 0)
     *
     * @note    Same as hsum but divides by idx at the end
     * @note    AtomicAdd used → safe for multi-block
     */
    __global__ void reduce_mean_kernel(const f4x32* __restrict__ Xx, f32* __restrict__ out, size_t idx) {
        __shared__ f32 shared[WARP_SIZE];

        const size_t tid    = threadIdx.x;
        const int warp_id   = tid / WARP_SIZE;
        const int lane_id   = tid % WARP_SIZE;

        f32 val = 0.0f;
        CXM_CUDA_LOOP(i, idx / 4) {
            val += Xx[i].x + Xx[i].y + Xx[i].z + Xx[i].w;
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid2 = global_thread_id();
        if (tid2 < tail) val += reinterpret_cast<const f32*>(Xx)[base + tid2];

        val = warp_reduce_sum(val);
        if (lane_id == 0) shared[warp_id] = val;
        __syncthreads();

        val = (tid < blockDim.x / WARP_SIZE) ? shared[lane_id] : 0.0f;
        if (warp_id == 0) val = warp_reduce_sum(val);
        if (tid == 0) atomicAdd(out, val);
    }
    /**
     * @brief   Computes in-place softmax over single vector: Xz[i] = exp(X[i] - max) / sum(exp(X - max))
     * @param   Xx      Input array (float4 view)
     * @param   Xz      Output array (float4 view, in-place possible but separate recommended)
     * @param   idx     Number of elements
     *
     * @note    Three-pass algorithm:
     *            1. Find max (reduce_hmax)
     *            2. Compute exp(X - max) and sum
     *            3. Normalize by 1/sum
     * @note    Uses warp shuffle + shared memory + broadcast
     * @note    For multi-row (per-batch) softmax use different kernel
     * @note    Numerical stability via max subtraction
     */
    __global__ void reduce_softmax_kernel(const f4x32* __restrict__ Xx, f4x32* __restrict__ Xz, size_t idx) {
        __shared__ f32 shared[WARP_SIZE];

        const size_t tid    = threadIdx.x;
        const int warp_id   = tid / WARP_SIZE;
        const int lane_id   = tid % WARP_SIZE;

        // adım 1: hmax
        f32 max_val = -CUDART_INF_F;
        CXM_CUDA_LOOP(i, idx / 4) {
            max_val = fmaxf(max_val, fmaxf(fmaxf(Xx[i].x, Xx[i].y), fmaxf(Xx[i].z, Xx[i].w)));
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid2 = global_thread_id();
        if (tid2 < tail) max_val = fmaxf(max_val, reinterpret_cast<const f32*>(Xx)[base + tid2]);

        max_val = warp_reduce_max(max_val);
        if (lane_id == 0) shared[warp_id] = max_val;
        __syncthreads();
        max_val = (tid < blockDim.x / WARP_SIZE) ? shared[lane_id] : -CUDART_INF_F;
        if (warp_id == 0) max_val = warp_reduce_max(max_val);
        max_val = __shfl_sync(0xffffffff, max_val, 0);
        __syncthreads();

        // adım 2: exp ve hsum
        f32 sum_val = 0.0f;
        CXM_CUDA_LOOP(i, idx / 4) {
            Xz[i] = {expf(Xx[i].x - max_val), expf(Xx[i].y - max_val),
                     expf(Xx[i].z - max_val), expf(Xx[i].w - max_val)};
            sum_val += Xz[i].x + Xz[i].y + Xz[i].z + Xz[i].w;
        }
        if (tid2 < tail) {
            f32& val = reinterpret_cast<f32*>(Xz)[base + tid2];
            val = expf(reinterpret_cast<const f32*>(Xx)[base + tid2] - max_val);
            sum_val += val;
        }

        sum_val = warp_reduce_sum(sum_val);
        if (lane_id == 0) shared[warp_id] = sum_val;
        __syncthreads();
        sum_val = (tid < blockDim.x / WARP_SIZE) ? shared[lane_id] : 0.0f;
        if (warp_id == 0) sum_val = warp_reduce_sum(sum_val);
        sum_val = __shfl_sync(0xffffffff, sum_val, 0);
        __syncthreads();

        // adım 3: normalize
        const f32 rcp_sum = 1.0f / sum_val;
        CXM_CUDA_LOOP(i, idx / 4) {
            Xz[i] = {Xz[i].x * rcp_sum, Xz[i].y * rcp_sum,
                     Xz[i].z * rcp_sum, Xz[i].w * rcp_sum};
        }
        if (tid2 < tail) reinterpret_cast<f32*>(Xz)[base + tid2] *= rcp_sum;
    }
    /**
     * @brief   Computes variance: out = sum((X[i] - mean)²) / idx
     * @param   Xx      Input array
     * @param   mean    Pre-computed mean value
     * @param   out     Single-element output
     * @param   idx     Number of elements
     *
     * @note    Uses FMA for squared difference
     * @note    AtomicAdd used → safe for multi-block
     */
    __global__ void reduce_var_kernel(const f4x32* __restrict__ Xx, f32 mean, f32* __restrict__ out, size_t idx) {
        __shared__ f32 shared[WARP_SIZE];

        const size_t tid    = threadIdx.x;
        const int warp_id   = tid / WARP_SIZE;
        const int lane_id   = tid % WARP_SIZE;

        f32 val = 0.0f;
        CXM_CUDA_LOOP(i, idx / 4) {
            const f32 dx = Xx[i].x - mean; val += dx * dx;
            const f32 dy = Xx[i].y - mean; val += dy * dy;
            const f32 dz = Xx[i].z - mean; val += dz * dz;
            const f32 dw = Xx[i].w - mean; val += dw * dw;
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid2 = global_thread_id();
        if (tid2 < tail) {
            const f32 d = reinterpret_cast<const f32*>(Xx)[base + tid2] - mean;
            val += d * d;
        }

        val = warp_reduce_sum(val);
        if (lane_id == 0) shared[warp_id] = val;
        __syncthreads();

        val = (tid < blockDim.x / WARP_SIZE) ? shared[lane_id] : 0.0f;
        if (warp_id == 0) val = warp_reduce_sum(val);
        if (tid == 0) atomicAdd(out, val);
    }
    /**
     * @brief   Computes Euclidean norm: out = √(sum(X[i]²))
     * @param   Xx      Input array
     * @param   out     Single-element output
     * @param   idx     Number of elements
     *
     * @note    Uses FMA for inner product accumulation
     * @note    AtomicAdd used → safe for multi-block
     */
    __global__ void reduce_norm_kernel(const f4x32* __restrict__ Xx, f32* __restrict__ out, size_t idx) {
        __shared__ f32 shared[WARP_SIZE];

        const size_t tid    = threadIdx.x;
        const int warp_id   = tid / WARP_SIZE;
        const int lane_id   = tid % WARP_SIZE;

        f32 val = 0.0f;
        CXM_CUDA_LOOP(i, idx / 4) {
            val += Xx[i].x * Xx[i].x + Xx[i].y * Xx[i].y +
                   Xx[i].z * Xx[i].z + Xx[i].w * Xx[i].w;
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid2 = global_thread_id();
        if (tid2 < tail) {
            const f32 v = reinterpret_cast<const f32*>(Xx)[base + tid2];
            val += v * v;
        }

        val = warp_reduce_sum(val);
        if (lane_id == 0) shared[warp_id] = val;
        __syncthreads();

        val = (tid < blockDim.x / WARP_SIZE) ? shared[lane_id] : 0.0f;
        if (warp_id == 0) val = warp_reduce_sum(val);
        if (tid == 0) atomicAdd(out, val);
    }
    /**
     * @brief   Reduces along the innermost dimension: Z[outer, after] = sum(X[outer, :, after])
     * @param   Xx      Input tensor (flattened)
     * @param   Xz      Output tensor (flattened, size outer × after)
     * @param   outer   Outer dimension size
     * @param   inner   Dimension to reduce (sum over this axis)
     * @param   after   Innermost dimension size
     *
     * @note    Each block processes one (outer_idx, after_idx) pair
     * @note    Inner loop uses thread striding (no shared memory needed)
     * @note    Single-thread write per output element
     * @note    Suitable for batch/channel reductions in CNNs/transformers
     */
    __global__ void reduce_sum_dim_kernel(
        const f32* __restrict__ Xx,
        f32* __restrict__ Xz,
        size_t outer,
        size_t inner,
        size_t after)
    {
        __shared__ f32 shared[WARP_SIZE];

        const size_t tid      = threadIdx.x;
        const int    warp_id  = tid / WARP_SIZE;
        const int    lane_id  = tid % WARP_SIZE;

        // her block bir (outer_idx, after_idx) çiftini işler
        const size_t after_idx = blockIdx.x % after;
        const size_t outer_idx = blockIdx.x / after;

        if (outer_idx >= outer) return;

        // inner boyunca topla
        f32 val = 0.0f;
        for (size_t i = tid; i < inner; i += blockDim.x) {
            val += Xx[outer_idx * inner * after + i * after + after_idx];
        }

        val = warp_reduce_sum(val);
        if (lane_id == 0) shared[warp_id] = val;
        __syncthreads();

        val = (tid < blockDim.x / WARP_SIZE) ? shared[lane_id] : 0.0f;
        if (warp_id == 0) val = warp_reduce_sum(val);
        if (tid == 0) Xz[outer_idx * after + after_idx] = val;
    }
} // namespace cortex::_fw::cuda::kernels

#endif // CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_KERNEL_CUH