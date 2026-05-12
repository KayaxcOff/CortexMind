//
// Created by muham on 12.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_DISPATCH_REDUCE_OPERATIONS_HPP
#define CORTEXMIND_FRAMEWORK_DISPATCH_REDUCE_OPERATIONS_HPP

#include <CortexMind/framework/Storage/stor.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/reduce.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE

namespace cortex::_fw::disp {
    /**
     * @brief Dispatch manager for reduction (fold) operations.
     *
     * Routes reduction requests to either AVX2 (CPU) or CUDA backend based on
     * the currently selected device. Uses `cuda::ReduceOp` for GPU operations.
     */
    class Reduce {
    public:
        /**
         * @brief Constructs a Reduce dispatcher for the specified device.
         *
         * @param _d_type Target computation device (HOST or CUDA)
         */
        explicit Reduce(sys::DeviceType _d_type);
        ~Reduce();

        /**
         * @brief Changes the active device for subsequent reduction operations.
         *
         * @param _d_type New target device
         */
        void SetDevice(sys::DeviceType _d_type);

        [[nodiscard]]
        f32 sum(const TensorStorage* __restrict x, size_t N);
        [[nodiscard]]
        f32 mean(const TensorStorage* __restrict x, size_t N);
        [[nodiscard]]
        f32 var(const TensorStorage* __restrict x, size_t N);
        [[nodiscard]]
        f32 std(const TensorStorage* __restrict x, size_t N);
        [[nodiscard]]
        f32 min(const TensorStorage* __restrict x, size_t N) const;
        [[nodiscard]]
        f32 max(const TensorStorage* __restrict x, size_t N) const;
        [[nodiscard]]
        f32 norm1(const TensorStorage* __restrict x, size_t N) const;
        [[nodiscard]]
        f32 norm2(const TensorStorage* __restrict x, size_t N) const;
        [[nodiscard]]
        f32 dot(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, size_t N) const;
    private:
        sys::DeviceType d_type;

        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp op;
        #endif //#if CXM_IS_CUDA_AVAILABLE
    };
} //namespace cortex::_fw::disp

#endif //CORTEXMIND_FRAMEWORK_DISPATCH_REDUCE_OPERATIONS_HPP