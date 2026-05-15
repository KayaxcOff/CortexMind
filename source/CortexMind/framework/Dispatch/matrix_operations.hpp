//
// Created by muham on 15.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_DISPATCH_MATRIX_OPERATIONS_HPP
#define CORTEXMIND_FRAMEWORK_DISPATCH_MATRIX_OPERATIONS_HPP

#include <CortexMind/framework/Storage/stor.hpp>
#include <CortexMind/framework/Tools/broadcast_kind.hpp>
#include <vector>

namespace cortex::_fw::disp {
    /**
     * @brief Dispatch manager for matrix and element-wise binary operations.
     *
     * Handles two cases:
     * - Same shape: routes directly to matrix backend (AVX2 or CUDA)
     * - Different shape: classifies broadcast kind, routes to broadcast backend
     *
     * Supported broadcast kinds:
     * - None → matrix_t / cuda::Matrix
     * - Row  → avx2::Broadcast::row_* / cuda::Broadcast::row_*
     * - Col  → avx2::Broadcast::col_* / cuda::Broadcast::col_*
     * - General → scalar fallback via BroadcastInfo stride walk
     */
    class Matrix {
    public:
        explicit Matrix(sys::DeviceType _d_type);
        ~Matrix();

        void SetDevice(sys::DeviceType _d_type);

        // ---------------------------------------------------------- //
        //  Element-wise binary ops — handles broadcast automatically  //
        // ---------------------------------------------------------- //

        void add(const TensorStorage* Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                 const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                 TensorStorage* Xz,       const std::vector<i64>& shape_z, const std::vector<i64>& stride_z) const;

        void sub(const TensorStorage* Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                 const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                 TensorStorage* Xz,       const std::vector<i64>& shape_z, const std::vector<i64>& stride_z) const;

        void mul(const TensorStorage* Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                 const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                 TensorStorage* Xz,       const std::vector<i64>& shape_z, const std::vector<i64>& stride_z) const;

        void div(const TensorStorage* Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                 const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                 TensorStorage* Xz,       const std::vector<i64>& shape_z, const std::vector<i64>& stride_z) const;

        // ---------------------------------------------------------- //
        //  In-place binary ops                                         //
        // ---------------------------------------------------------- //

        void add(TensorStorage* Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                 const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y) const;

        void sub(TensorStorage* Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                 const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y) const;

        void mul(TensorStorage* Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                 const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y) const;

        void div(TensorStorage* Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                 const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y) const;

        // ---------------------------------------------------------- //
        //  Matrix multiplication                                       //
        // ---------------------------------------------------------- //

        /**
         * @brief General matrix multiplication: Xz = Xx @ Xy
         *
         * @param xN Row count of Xx
         * @param yN Shared dimension (cols of Xx, rows of Xy)
         * @param zN Col count of Xy
         */
        void matmul(const TensorStorage* Xx, const TensorStorage* Xy,
                    TensorStorage* Xz,
                    size_t xN, size_t yN, size_t zN) const;
    private:
        sys::DeviceType d_type;

        // Internal dispatch helpers
        void dispatch_add(const f32* x, const f32* y, f32* z,
                          const std::vector<i64>& sx, const std::vector<i64>& sy,
                          const std::vector<i64>& shape_x, const std::vector<i64>& shape_y,
                          const std::vector<i64>& shape_z, const std::vector<i64>& stride_z,
                          BroadcastKind kind) const;

        void dispatch_sub(const f32* x, const f32* y, f32* z,
                          const std::vector<i64>& sx, const std::vector<i64>& sy,
                          const std::vector<i64>& shape_x, const std::vector<i64>& shape_y,
                          const std::vector<i64>& shape_z, const std::vector<i64>& stride_z,
                          BroadcastKind kind) const;

        void dispatch_mul(const f32* x, const f32* y, f32* z,
                          const std::vector<i64>& sx, const std::vector<i64>& sy,
                          const std::vector<i64>& shape_x, const std::vector<i64>& shape_y,
                          const std::vector<i64>& shape_z, const std::vector<i64>& stride_z,
                          BroadcastKind kind) const;

        void dispatch_div(const f32* x, const f32* y, f32* z,
                          const std::vector<i64>& sx, const std::vector<i64>& sy,
                          const std::vector<i64>& shape_x, const std::vector<i64>& shape_y,
                          const std::vector<i64>& shape_z, const std::vector<i64>& stride_z,
                          BroadcastKind kind) const;
    };
} //namespace cortex::_fw::disp

#endif //CORTEXMIND_FRAMEWORK_DISPATCH_MATRIX_OPERATIONS_HPP