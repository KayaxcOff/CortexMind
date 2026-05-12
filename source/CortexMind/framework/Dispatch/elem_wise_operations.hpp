//
// Created by muham on 12.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_DISPATCH_ELEM_WISE_OPERATIONS_HPP
#define CORTEXMIND_FRAMEWORK_DISPATCH_ELEM_WISE_OPERATIONS_HPP

#include <CortexMind/framework/Storage/stor.hpp>

namespace cortex::_fw::disp {
    /**
     * @brief Dispatch manager for element-wise mathematical operations.
     *
     * This class routes unary mathematical operations to the appropriate
     * backend implementation depending on the active device (Host or CUDA).
     */
    class ElementWise {
    public:
        /**
         * @brief Constructs an ElementWise dispatcher for the specified device.
         *
         * @param _d_type Target device type (HOST or CUDA)
         */
        explicit ElementWise(sys::DeviceType _d_type);
        ~ElementWise();

        /**
         * @brief Changes the active device for subsequent operations.
         */
        void SetDevice(sys::DeviceType _d_type);

        /**
         * @brief Element-wise power: `Z[i] = X[i] ^ exp`
         */
        void pow(const TensorStorage* __restrict Xx, f32 exp, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief Element-wise square root: `Z[i] = sqrt(X[i])`
         */
        void sqrt(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief Element-wise natural logarithm: `Z[i] = log(X[i])`
         */
        void log(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief Element-wise exponential: `Z[i] = exp(X[i])`
         */
        void exp(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief Element-wise absolute value: `Z[i] = |X[i]|`
         */
        void abs(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
    private:
        sys::DeviceType d_type;
    };
} //namespace cortex::_fw::disp

#endif //CORTEXMIND_FRAMEWORK_DISPATCH_ELEM_WISE_OPERATIONS_HPP