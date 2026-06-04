//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_REDUCE_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_REDUCE_CUH

#include <CortexMind/framework/Tools/cuda.cuh>
#include <CortexMind/framework/Tools/loop.hpp>
#include <CortexMind/framework/Tools/types.hpp>
#include <cmath>

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief Performs warp-level sum reduction using shuffle instructions.
     *
     * @param val Input value per thread.
     * @return Sum of all values in the warp (valid only for lane 0).
     */
    __device__ inline f32 warp_reduce_sum(f32 val) {
        val += shfl::down(val, 16);
        val += shfl::down(val, 8);
        val += shfl::down(val, 4);
        val += shfl::down(val, 2);
        val += shfl::down(val, 1);
        return val;
    }

    /**
     * @brief Performs warp-level maximum reduction using shuffle instructions.
     *
     * @param val Input value per thread.
     * @return Maximum value in the warp (valid only for lane 0).
     */
    __device__ inline f32 warp_reduce_max(f32 val) {
        val = fmaxf(val, shfl::down(val, 16));
        val = fmaxf(val, shfl::down(val, 8));
        val = fmaxf(val, shfl::down(val, 4));
        val = fmaxf(val, shfl::down(val, 2));
        val = fmaxf(val, shfl::down(val, 1));
        return val;
    }

    struct KeyValuePair {
        f32 value;
        i32 index;
    };

    __device__ inline KeyValuePair warp_reduce_argmax(KeyValuePair kv) {
        KeyValuePair target;
        #pragma unroll
        for (i32 offset = 16; offset > 0; offset /= 2) {
            target.value = shfl::down(kv.value, offset);
            target.index = shfl::down(kv.index, offset);
            if (target.value > kv.value || (target.value == kv.value && target.index < kv.index)) {
                kv = target;
            }
        }
        return kv;
    }

    __device__ inline KeyValuePair warp_reduce_argmin(KeyValuePair kv) {
        KeyValuePair target;
        #pragma unroll
        for (i32 offset = 16; offset > 0; offset /= 2) {
            target.value = shfl::down(kv.value, offset);
            target.index = shfl::down(kv.index, offset);
            if (target.value < kv.value || (target.value == kv.value && target.index < kv.index)) {
                kv = target;
            }
        }
        return kv;
    }

    /**
     * @brief CUDA kernel for parallel sum reduction.
     *
     * Computes `sum(Xx[0..N-1])` and adds the result to `*Xz` using atomic operation.
     *
     * @param Xx Input array
     * @param Xz Output pointer (single float, accumulated via atomicAdd)
     * @param N  Number of elements
     *
     * @note Uses warp shuffle + shared memory reduction.
     * @note Multiple blocks can be launched; result is accumulated atomically.
     */
    __global__ void ReduceSum(const f32* __restrict Xx, f32* __restrict Xz, const size_t N) {
        extern __shared__ f32 sdata[];

        const size_t tid = threadIdx.x;
        f32 local = 0.0f;

        CXM_CUDA_LOOP_1D(i, N) {
            local += Xx[i];
        }

        local = warp_reduce_sum(local);

        if (tid % WARP_SIZE == 0) {
            sdata[tid / WARP_SIZE] = local;
        }

        SynchronizeThreads();

        const size_t warp_count = blockDim.x / WARP_SIZE;
        if (tid < warp_count) {
            f32 warp_val = sdata[tid];
            warp_val = warp_reduce_sum(warp_val);
            if (tid == 0) {
                atomic::add<f32>(Xz, warp_val);
            }
        }
    }

    /**
     * @brief CUDA kernel for parallel variance reduction.
     *
     * Computes `sum((x[i] - mean)^2)` for i in [0, N) and adds the result
     * to `*Xz` using atomic operation.
     *
     * @param Xx  Input array
     * @param mean Pre-computed mean value
     * @param Xz  Output pointer (single float, accumulated via atomicAdd)
     * @param N   Number of elements
     *
     * @note Used as the second step in variance / standard deviation calculation.
     */
    __global__ void ReduceVar(const f32* __restrict Xx, const f32 mean, f32* __restrict Xz, const size_t N) {
        extern __shared__ f32 sdata[];

        const size_t tid = threadIdx.x;
        f32 local = 0.0f;

        CXM_CUDA_LOOP_1D(i, N) {
            const f32 diff = Xx[i] - mean;
            local += diff * diff;
        }

        local = warp_reduce_sum(local);

        if (tid % WARP_SIZE == 0) {
            sdata[tid / WARP_SIZE] = local;
        }

        SynchronizeThreads();

        const size_t warp_count = blockDim.x / WARP_SIZE;
        if (tid < warp_count) {
            f32 warp_val = sdata[tid];
            warp_val = warp_reduce_sum(warp_val);
            if (tid == 0) {
                atomic::add<f32>(Xz, warp_val);
            }
        }
    }

    __global__ void ReduceSumFirstDim(const f32* __restrict__ Xx, f32* __restrict__ Xz, const size_t rows, const size_t cols) {
        extern __shared__ f32 sdata[];

        const size_t col = blockIdx.x;
        const size_t tid = threadIdx.x;

        if (col >= cols) return;

        f32 local = 0.0f;
        for (size_t r = tid; r < rows; r += blockDim.x) {
            local += Xx[r * cols + col];
        }

        // Warp reduce
        local = warp_reduce_sum(local);

        if (tid % WARP_SIZE == 0) {
            sdata[tid / WARP_SIZE] = local;
        }
        SynchronizeThreads();

        const size_t warp_count = blockDim.x / WARP_SIZE;
        if (tid < warp_count) {
            f32 val = sdata[tid];
            val = warp_reduce_sum(val);
            if (tid == 0) {
                Xz[col] = val;
            }
        }
    }

    __global__ void ReduceSumLastDim(const f32* __restrict__ Xx, f32* __restrict__ Xz, const size_t rows, const size_t cols) {
        extern __shared__ f32 sdata[];

        const size_t row = blockIdx.x;
        const size_t tid = threadIdx.x;

        if (row >= rows) return;

        f32 local = 0.0f;
        for (size_t c = tid; c < cols; c += blockDim.x) {
            local += Xx[row * cols + c];
        }

        local = warp_reduce_sum(local);

        if (tid % WARP_SIZE == 0) {
            sdata[tid / WARP_SIZE] = local;
        }
        SynchronizeThreads();

        const size_t warp_count = blockDim.x / WARP_SIZE;
        if (tid < warp_count) {
            f32 val = sdata[tid];
            val = warp_reduce_sum(val);
            if (tid == 0) {
                Xz[row] = val;
            }
        }
    }

    /**
     * @brief Reduce sum along a dimension (template-based, compile-time block size).
     *
     * Each block handles exactly one output element → no atomic operations needed.
     */
    template <size_t BlockSize>
    __global__ void ReduceSumDim(const f32* __restrict Xx, f32* __restrict Xz,
                                 size_t outer_size, size_t dim_size, size_t inner_size) {
        __shared__ f32 sdata[BlockSize / WARP_SIZE];

        size_t out_idx = blockIdx.x;
        if (out_idx >= outer_size * inner_size) return;

        size_t outer_coord = out_idx / inner_size;
        size_t inner_coord = out_idx % inner_size;
        size_t base_offset = outer_coord * (dim_size * inner_size) + inner_coord;

        // Local reduction across dimension
        f32 local_sum = 0.0f;
        for (size_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
            local_sum += Xx[base_offset + d * inner_size];
        }

        // Warp reduce
        local_sum = warp_reduce_sum(local_sum);

        const size_t tid = threadIdx.x;
        const size_t lane = tid % WARP_SIZE;
        const size_t warp_id = tid / WARP_SIZE;

        // Warp leads write to shared memory
        if (lane == 0) {
            sdata[warp_id] = local_sum;
        }

        SynchronizeThreads();

        // Block reduce (first warp processes all warp results)
        if (warp_id == 0) {
            f32 val = (tid < (BlockSize / WARP_SIZE)) ? sdata[tid] : 0.0f;
            val = warp_reduce_sum(val);

            if (tid == 0) {
                Xz[out_idx] = val;  // Direct write (no atomic needed!)
            }
        }
    }

    /**
     * @brief Reduce mean along a dimension.
     */
    template <size_t BlockSize>
    __global__ void ReduceMeanDim(const f32* __restrict Xx, f32* __restrict Xz,
                                  size_t outer_size, size_t dim_size, size_t inner_size) {
        __shared__ f32 sdata[BlockSize / WARP_SIZE];

        size_t out_idx = blockIdx.x;
        if (out_idx >= outer_size * inner_size) return;

        size_t outer_coord = out_idx / inner_size;
        size_t inner_coord = out_idx % inner_size;
        size_t base_offset = outer_coord * (dim_size * inner_size) + inner_coord;

        f32 local_sum = 0.0f;
        for (size_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
            local_sum += Xx[base_offset + d * inner_size];
        }

        local_sum = warp_reduce_sum(local_sum);

        const size_t tid = threadIdx.x;
        const size_t lane = tid % WARP_SIZE;
        const size_t warp_id = tid / WARP_SIZE;

        if (lane == 0) {
            sdata[warp_id] = local_sum;
        }

        SynchronizeThreads();

        if (warp_id == 0) {
            f32 val = (tid < (BlockSize / WARP_SIZE)) ? sdata[tid] : 0.0f;
            val = warp_reduce_sum(val);

            if (tid == 0) {
                Xz[out_idx] = val / static_cast<f32>(dim_size);
            }
        }
    }

    /**
     * @brief Reduce variance along a dimension (requires pre-computed means).
     */
    template <size_t BlockSize>
    __global__ void ReduceVarDim(const f32* __restrict Xx, f32* __restrict Xz,
                                 const f32* __restrict means,
                                 size_t outer_size, size_t dim_size, size_t inner_size) {
        __shared__ f32 sdata[BlockSize / WARP_SIZE];

        size_t out_idx = blockIdx.x;
        if (out_idx >= outer_size * inner_size) return;

        size_t outer_coord = out_idx / inner_size;
        size_t inner_coord = out_idx % inner_size;
        size_t base_offset = outer_coord * (dim_size * inner_size) + inner_coord;

        const f32 mean = means[out_idx];

        f32 local_var = 0.0f;
        for (size_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
            f32 diff = Xx[base_offset + d * inner_size] - mean;
            local_var += diff * diff;
        }

        local_var = warp_reduce_sum(local_var);

        const size_t tid = threadIdx.x;
        const size_t lane = tid % WARP_SIZE;
        const size_t warp_id = tid / WARP_SIZE;

        if (lane == 0) {
            sdata[warp_id] = local_var;
        }

        SynchronizeThreads();

        if (warp_id == 0) {
            f32 val = (tid < (BlockSize / WARP_SIZE)) ? sdata[tid] : 0.0f;
            val = warp_reduce_sum(val);

            if (tid == 0) {
                Xz[out_idx] = val / static_cast<f32>(dim_size);
            }
        }
    }

    /**
     * @brief Reduce min along a dimension.
     */
    template <size_t BlockSize>
    __global__ void ReduceMinDim(const f32* __restrict Xx, f32* __restrict Xz,
                                 size_t outer_size, size_t dim_size, size_t inner_size) {
        __shared__ f32 sdata[BlockSize / WARP_SIZE];

        size_t out_idx = blockIdx.x;
        if (out_idx >= outer_size * inner_size) return;

        size_t outer_coord = out_idx / inner_size;
        size_t inner_coord = out_idx % inner_size;
        size_t base_offset = outer_coord * (dim_size * inner_size) + inner_coord;

        f32 local_min = INFINITY;
        for (size_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
            local_min = fminf(local_min, Xx[base_offset + d * inner_size]);
        }

        // Warp reduce min
        local_min = fminf(local_min, shfl::down(local_min, 16));
        local_min = fminf(local_min, shfl::down(local_min, 8));
        local_min = fminf(local_min, shfl::down(local_min, 4));
        local_min = fminf(local_min, shfl::down(local_min, 2));
        local_min = fminf(local_min, shfl::down(local_min, 1));

        const size_t tid = threadIdx.x;
        const size_t lane = tid % WARP_SIZE;
        const size_t warp_id = tid / WARP_SIZE;

        if (lane == 0) {
            sdata[warp_id] = local_min;
        }

        SynchronizeThreads();

        if (warp_id == 0) {
            const size_t actual_warps = blockDim.x / WARP_SIZE;
            f32 val = (tid < actual_warps) ? sdata[tid] : INFINITY;
            val = fminf(val, shfl::down(val, 16));
            val = fminf(val, shfl::down(val, 8));
            val = fminf(val, shfl::down(val, 4));
            val = fminf(val, shfl::down(val, 2));
            val = fminf(val, shfl::down(val, 1));

            if (tid == 0) {
                Xz[out_idx] = val;
            }
        }
    }

    /**
     * @brief Reduce max along a dimension.
     */
    template <size_t BlockSize>
    __global__ void ReduceMaxDim(const f32* __restrict Xx, f32* __restrict Xz,
                                 size_t outer_size, size_t dim_size, size_t inner_size) {
        __shared__ f32 sdata[BlockSize / WARP_SIZE];

        size_t out_idx = blockIdx.x;
        if (out_idx >= outer_size * inner_size) return;

        size_t outer_coord = out_idx / inner_size;
        size_t inner_coord = out_idx % inner_size;
        size_t base_offset = outer_coord * (dim_size * inner_size) + inner_coord;

        f32 local_max = -INFINITY;
        for (size_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
            local_max = fmaxf(local_max, Xx[base_offset + d * inner_size]);
        }

        // Warp reduce max
        local_max = fmaxf(local_max, shfl::down(local_max, 16));
        local_max = fmaxf(local_max, shfl::down(local_max, 8));
        local_max = fmaxf(local_max, shfl::down(local_max, 4));
        local_max = fmaxf(local_max, shfl::down(local_max, 2));
        local_max = fmaxf(local_max, shfl::down(local_max, 1));

        const size_t tid = threadIdx.x;
        const size_t lane = tid % WARP_SIZE;
        const size_t warp_id = tid / WARP_SIZE;

        if (lane == 0) {
            sdata[warp_id] = local_max;
        }

        SynchronizeThreads();

        if (warp_id == 0) {
            const size_t actual_warps = blockDim.x / WARP_SIZE;
            f32 val = (tid < actual_warps) ? sdata[tid] : -INFINITY;
            val = fmaxf(val, shfl::down(val, 16));
            val = fmaxf(val, shfl::down(val, 8));
            val = fmaxf(val, shfl::down(val, 4));
            val = fmaxf(val, shfl::down(val, 2));
            val = fmaxf(val, shfl::down(val, 1));

            if (tid == 0) {
                Xz[out_idx] = val;
            }
        }
    }

    /**
     * @brief Global ArgMax kernel.
     * @param Xz Output index pointer (single integer)
     */
    __global__ void ArgMax(const f32* __restrict Xx, i32* __restrict Xz, const size_t N) {
        extern __shared__ KeyValuePair kv_sdata[];

        const size_t tid = threadIdx.x;
        KeyValuePair local = {-INFINITY, -1};

        CXM_CUDA_LOOP_1D(i, N) {
            if (Xx[i] > local.value) {
                local.value = Xx[i];
                local.index = static_cast<int>(i);
            }
        }

        local = warp_reduce_argmax(local);

        if (tid % WARP_SIZE == 0) {
            kv_sdata[tid / WARP_SIZE] = local;
        }

        SynchronizeThreads();

        const size_t warp_count = blockDim.x / WARP_SIZE;
        if (tid < warp_count) {
            KeyValuePair warp_val = kv_sdata[tid];
            warp_val = warp_reduce_argmax(warp_val);
            if (tid == 0) {
                static_assert(sizeof(unsigned long long) == sizeof(KeyValuePair), "Size mismatch for atomic");
                if(blockIdx.x == 0) {
                    *Xz = warp_val.index;
                }
            }
        }
    }

    /**
     * @brief CUDA kernel for parallel global ArgMin reduction.
     *
     * Finds the minimum value's index in the entire array Xx[0..N-1]
     * and writes it to *Xz.
     *
     * @param Xx Input array
     * @param Xz Output pointer (single integer for the global minimum index)
     * @param N  Number of elements
     *
     * @note Designed for a single-block execution or coordinated multi-block.
     */
    __global__ void ArgMin(const f32* __restrict Xx, int* __restrict Xz, const size_t N) {
        extern __shared__ char shared_mem[];
        KeyValuePair* kv_sdata = reinterpret_cast<KeyValuePair*>(shared_mem);

        const size_t tid = threadIdx.x;
        KeyValuePair local = {INFINITY, -1};

        // Grid-stride loop ile tüm elemanları tara
        CXM_CUDA_LOOP_1D(i, N) {
            f32 val = Xx[i];
            if (val < local.value) {
                local.value = val;
                local.index = static_cast<int>(i);
            }
        }

        local = warp_reduce_argmin(local);

        if (tid % WARP_SIZE == 0) {
            kv_sdata[tid / WARP_SIZE] = local;
        }

        SynchronizeThreads();

        const size_t warp_count = blockDim.x / WARP_SIZE;
        if (tid < warp_count) {
            KeyValuePair warp_val = kv_sdata[tid];
            warp_val = warp_reduce_argmin(warp_val);

            if (tid == 0) {
                if (blockIdx.x == 0) {
                    *Xz = warp_val.index;
                }
            }
        }
    }

    /**
     * @brief Reduce ArgMax along a dimension.
     * @param Xz Output index array (type: int)
     */
    template <size_t BlockSize>
    __global__ void ReduceArgMaxDim(const f32* __restrict Xx, int* __restrict Xz,
                                    size_t outer_size, size_t dim_size, size_t inner_size) {
        __shared__ KeyValuePair sdata[BlockSize / WARP_SIZE];

        size_t out_idx = blockIdx.x;
        if (out_idx >= outer_size * inner_size) return;

        size_t outer_coord = out_idx / inner_size;
        size_t inner_coord = out_idx % inner_size;
        size_t base_offset = outer_coord * (dim_size * inner_size) + inner_coord;

        KeyValuePair local_max = {-INFINITY, -1};
        for (size_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
            f32 val = Xx[base_offset + d * inner_size];
            if (val > local_max.value) {
                local_max.value = val;
                local_max.index = static_cast<i32>(d);
            }
        }

        local_max = warp_reduce_argmax(local_max);

        const size_t tid = threadIdx.x;
        const size_t lane = tid % WARP_SIZE;
        const size_t warp_id = tid / WARP_SIZE;

        if (lane == 0) {
            sdata[warp_id] = local_max;
        }

        SynchronizeThreads();

        if (warp_id == 0) {
            KeyValuePair val = (tid < (BlockSize / WARP_SIZE)) ? sdata[tid] : KeyValuePair{-INFINITY, -1};
            val = warp_reduce_argmax(val);

            if (tid == 0) {
                Xz[out_idx] = val.index;
            }
        }
    }

    /**
     * @brief Reduce ArgMin along a dimension.
     */
    template <size_t BlockSize>
    __global__ void ReduceArgMinDim(const f32* __restrict Xx, int* __restrict Xz,
                                    size_t outer_size, size_t dim_size, size_t inner_size) {
        __shared__ KeyValuePair sdata[BlockSize / WARP_SIZE];

        size_t out_idx = blockIdx.x;
        if (out_idx >= outer_size * inner_size) return;

        size_t outer_coord = out_idx / inner_size;
        size_t inner_coord = out_idx % inner_size;
        size_t base_offset = outer_coord * (dim_size * inner_size) + inner_coord;

        KeyValuePair local_min = {INFINITY, -1};
        for (size_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
            f32 val = Xx[base_offset + d * inner_size];
            if (val < local_min.value) {
                local_min.value = val;
                local_min.index = static_cast<int>(d);
            }
        }

        local_min = warp_reduce_argmin(local_min);

        const size_t tid = threadIdx.x;
        const size_t lane = tid % WARP_SIZE;
        const size_t warp_id = tid / WARP_SIZE;

        if (lane == 0) {
            sdata[warp_id] = local_min;
        }

        SynchronizeThreads();

        if (warp_id == 0) {
            KeyValuePair val = (tid < (BlockSize / WARP_SIZE)) ? sdata[tid] : KeyValuePair{INFINITY, -1}
            val = warp_reduce_argmin(val);

            if (tid == 0) {
                Xz[out_idx] = val.index;
            }
        }
    }

    /**
     * @brief Reduce L1 Norm (Sum of absolute values) along a dimension.
     */
    template <size_t BlockSize>
    __global__ void ReduceNorm1Dim(const f32* __restrict Xx, f32* __restrict Xz,
                                  size_t outer_size, size_t dim_size, size_t inner_size) {
        __shared__ f32 sdata[BlockSize / WARP_SIZE];

        size_t out_idx = blockIdx.x;
        if (out_idx >= outer_size * inner_size) return;

        size_t outer_coord = out_idx / inner_size;
        size_t inner_coord = out_idx % inner_size;
        size_t base_offset = outer_coord * (dim_size * inner_size) + inner_coord;

        f32 local_sum = 0.0f;
        for (size_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
            local_sum += fabsf(Xx[base_offset + d * inner_size]);
        }

        local_sum = warp_reduce_sum(local_sum);

        const size_t tid = threadIdx.x;
        const size_t lane = tid % WARP_SIZE;
        const size_t warp_id = tid / WARP_SIZE;

        if (lane == 0) {
            sdata[warp_id] = local_sum;
        }

        SynchronizeThreads();

        if (warp_id == 0) {
            f32 val = (tid < (BlockSize / WARP_SIZE)) ? sdata[tid] : 0.0f;
            val = warp_reduce_sum(val);

            if (tid == 0) {
                Xz[out_idx] = val;
            }
        }
    }

    /**
     * @brief Reduce L2 Norm (Square root of sum of squares) along a dimension.
     */
    template <size_t BlockSize>
    __global__ void ReduceNorm2Dim(const f32* __restrict Xx, f32* __restrict Xz,
                                  size_t outer_size, size_t dim_size, size_t inner_size) {
        __shared__ f32 sdata[BlockSize / WARP_SIZE];

        size_t out_idx = blockIdx.x;
        if (out_idx >= outer_size * inner_size) return;

        size_t outer_coord = out_idx / inner_size;
        size_t inner_coord = out_idx % inner_size;
        size_t base_offset = outer_coord * (dim_size * inner_size) + inner_coord;

        f32 local_squares = 0.0f;
        for (size_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
            f32 val = Xx[base_offset + d * inner_size];
            local_squares += val * val;
        }

        local_squares = warp_reduce_sum(local_squares);

        const size_t tid = threadIdx.x;
        const size_t lane = tid % WARP_SIZE;
        const size_t warp_id = tid / WARP_SIZE;

        if (lane == 0) {
            sdata[warp_id] = local_squares;
        }

        SynchronizeThreads();

        if (warp_id == 0) {
            f32 val = (tid < (BlockSize / WARP_SIZE)) ? sdata[tid] : 0.0f;
            val = warp_reduce_sum(val);

            if (tid == 0) {
                Xz[out_idx] = sqrtf(val);
            }
        }
    }

    /**
     * @brief Reduce Standard Deviation along a dimension (requires pre-computed means).
     */
    template <size_t BlockSize>
    __global__ void ReduceStdvDim(const f32* __restrict Xx, f32* __restrict Xz,
                                 const f32* __restrict means,
                                 size_t outer_size, size_t dim_size, size_t inner_size) {
        __shared__ f32 sdata[BlockSize / WARP_SIZE];

        size_t out_idx = blockIdx.x;
        if (out_idx >= outer_size * inner_size) return;

        size_t outer_coord = out_idx / inner_size;
        size_t inner_coord = out_idx % inner_size;
        size_t base_offset = outer_coord * (dim_size * inner_size) + inner_coord;

        const f32 mean = means[out_idx];

        f32 local_var = 0.0f;
        for (size_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
            f32 diff = Xx[base_offset + d * inner_size] - mean;
            local_var += diff * diff;
        }

        local_var = warp_reduce_sum(local_var);

        const size_t tid = threadIdx.x;
        const size_t lane = tid % WARP_SIZE;
        const size_t warp_id = tid / WARP_SIZE;

        if (lane == 0) {
            sdata[warp_id] = local_var;
        }

        SynchronizeThreads();

        if (warp_id == 0) {
            f32 val = (tid < (BlockSize / WARP_SIZE)) ? sdata[tid] : 0.0f;
            val = warp_reduce_sum(val);

            if (tid == 0) {
                Xz[out_idx] = sqrtf(val / static_cast<f32>(dim_size));
            }
        }
    }
} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_REDUCE_CUH