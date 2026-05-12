//
// Created by muham on 11.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_DISPATCH_SCALAR_OPERATIONS_HPP
#define CORTEXMIND_FRAMEWORK_DISPATCH_SCALAR_OPERATIONS_HPP

#include <CortexMind/framework/Storage/stor.hpp>

namespace cortex::_fw::disp {
    /**
     * @brief Dispatch manager for scalar (element + scalar) operations.
     *
     * This class acts as a high-level dispatcher that routes scalar arithmetic
     * operations to the appropriate backend implementation (CPU/AVX2 or CUDA)
     * depending on the selected device.
     */
    class Scalar {
    public:
        /**
         * @brief Constructs a Scalar dispatcher for the specified device.
         *
         * @param _d_type Target device type (HOST or CUDA)
         */
        explicit Scalar(sys::DeviceType _d_type);
        ~Scalar();

        /**
         * @brief Changes the active device for subsequent operations.
         *
         * @param _d_type New target device
         */
        void SetDevice(sys::DeviceType _d_type);

        /**
         * @brief out-place addition : Xz = Xx + value
         *
         * @param Xx        input Storage
         * @param value     input scalar value
         * @param Xz        output Storage
         * @param N         number of element
         */
        void add(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief out-place substitution : Xz = Xx - value
         *
         * @param Xx        input Storage
         * @param value     input scalar value
         * @param Xz        output Storage
         * @param N         number of element
         */
        void sub(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief out-place multiply : Xz = Xx * value
         *
         * @param Xx        input Storage
         * @param value     input scalar value
         * @param Xz        putput Storage
         * @param N         number of element
         */
        void mul(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief out-place division : Xz = Xx / value
         *
         * @param Xx        input Storage
         * @param value     input scalar value
         * @param Xz        output Storage
         * @param N         number of element
         */
        void div(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz, size_t N) const;

        /**
         * @brief in-place addition : Xx = Xx + value
         *
         * @param Xx        input and output Storage
         * @param value     input scalar value
         * @param N         number of element
         */
        void add(TensorStorage* Xx, f32 value, size_t N);
        /**
         * @brief in-place subscription : Xx = Xx - value
         *
         * @param Xx        input and output Storage
         * @param value     input scalar value
         * @param N         number of element
         */
        void sub(TensorStorage* Xx, f32 value, size_t N);
        /**
         * @brief in-place multiply : Xx = Xx * value
         *
         * @param Xx        input and output Storage
         * @param value     input scalar value
         * @param N         number of element
         */
        void mul(TensorStorage* Xx, f32 value, size_t N);
        /**
         * @brief in-place division : Xx = Xx / value
         *
         * @param Xx        input and output Storage
         * @param value     input scalar value
         * @param N         number of element
         */
        void div(TensorStorage* Xx, f32 value, size_t N);
    private:
        sys::DeviceType d_type;
    };
} //namespace cortex::_fw::disp

#endif //CORTEXMIND_FRAMEWORK_DISPATCH_SCALAR_OPERATIONS_HPP