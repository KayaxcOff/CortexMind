//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_CUH

#include <CortexMind/core/Tools/params.hpp>
#include <memory>

namespace cortex::_fw::cuda {
    /**
     * @brief   RAII wrapper for CUDA reduction operations
     *
     * Lazily allocates a single-element device buffer for reduction results.
     * Launches appropriate kernels and synchronizes for immediate results.
     *
     * Usage example:
     * @code
     *     cuda::reduce_t reducer;
     *     float sum = reducer.hsum(d_data, n);
     *     float max_val = reducer.hmax(d_data, n);
     *     reducer.softmax(d_logits, d_probs, n);
     * @endcode
     */
    struct reduce_t {
        reduce_t();
        ~reduce_t() = default;

        /**
         * @brief   Computes sum of all elements: sum(X[0..idx-1])
         * @param   Xx      Device input array
         * @param   idx     Number of elements
         * @return  Sum as host float
         */
        f32  hsum(const f32* Xx, size_t idx);
        /**
         * @brief   Computes maximum value
         * @param   Xx      Device input array
         * @param   idx     Number of elements
         * @return  Max value
         */
        f32  hmax(const f32* Xx, size_t idx);
        /**
         * @brief   Computes minimum value
         * @param   Xx      Device input array
         * @param   idx     Number of elements
         * @return  Max value
         */
        f32  hmin(const f32* Xx, size_t idx);
        /**
         * @brief   Computes arithmetic mean: sum(X) / idx
         * @param   Xx      Device input array
         * @param   idx     Number of elements (must be > 0)
         * @return  Mean value
         */
        f32  mean(const f32* Xx, size_t idx);
        /**
         * @brief   Computes in-place softmax over single vector
         * @param   Xx      Device input logits
         * @param   Xz      Device output probabilities (can be same as Xx)
         * @param   idx     Number of elements
         *
         * @note    Numerically stable (subtracts max before exp)
         * @note    Single-row softmax only
         */
        void softmax(const f32* __restrict Xx, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Computes variance: sum((X[i] - mean)²) / idx
         * @param   Xx      Device input array
         * @param   idx     Number of elements (must be > 0)
         * @return  Variance (unbiased estimator)
         *
         * @note    Internally computes mean first if not provided
         * @note    Uses FMA for better numerical stability
         */
        f32 var(const f32* Xx, size_t idx);
        /**
         * @brief   Computes Euclidean (L2) norm: √(sum(X[i]²))
         * @param   Xx      Device input array
         * @param   idx     Number of elements
         * @return  L2 norm value
         */
        f32 norm(const f32* Xx, size_t idx);
        /**
         * @brief   Computes sum of all elements (alias for hsum)
         * @param   Xx      Device input array
         * @param   idx     Number of elements
         * @return  Sum value
         */
        f32 sum(const f32* Xx, size_t idx);
        /**
         * @brief   Computes in-place softmax over single vector
         * @param   Xx      Device input logits
         * @param   Xz      Device output probabilities (can be same as Xx)
         * @param   idx     Number of elements
         *
         * @note    Numerically stable (subtracts max before exp)
         * @note    Single-row/vector only — for multi-row/batch use custom kernel
         */
        void sum_dim(const f32* __restrict__ Xx, f32* __restrict__ Xz, size_t outer, size_t inner, size_t after);

    private:
        struct cuda_deleter {
            void operator()(f32* ptr) const { cudaFree(ptr); }
        };
        std::unique_ptr<f32, cuda_deleter> d_tmp;
    };
} // namespace cortex::_fw::cuda

#endif // CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_CUH