//
// Created by muham on 16.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_MATRIX_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_MATRIX_HPP

#include <CortexMind/framework/Storage/stor.hpp>
#include <CortexMind/framework/Tools/tensor_meta.hpp>

namespace cortex::_fw::ix {
    /**
     * @brief Static dispatch for matrix and element-wise binary operations.
     *
     * Automatically detects broadcast kind from shapes and routes to the
     * correct backend (AVX2 or CUDA) and broadcast path (None/Row/Col/General).
     */
    struct MatrixOp {
        // -------------------------------------------------------- //
        //  Out-of-place                                             //
        // -------------------------------------------------------- //
        static void add(const TensorStorage* Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                        const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                        TensorStorage*       Xz, const std::vector<i64>& shape_z, const std::vector<i64>& stride_z);

        static void sub(const TensorStorage* Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                        const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                        TensorStorage*       Xz, const std::vector<i64>& shape_z, const std::vector<i64>& stride_z);

        static void mul(const TensorStorage* Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                        const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                        TensorStorage*       Xz, const std::vector<i64>& shape_z, const std::vector<i64>& stride_z);

        static void div(const TensorStorage* Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                        const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                        TensorStorage*       Xz, const std::vector<i64>& shape_z, const std::vector<i64>& stride_z);

        // -------------------------------------------------------- //
        //  In-place                                                 //
        // -------------------------------------------------------- //
        static void add(TensorStorage*       Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                        const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y);

        static void sub(TensorStorage*       Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                        const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y);

        static void mul(TensorStorage*       Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                        const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y);

        static void div(TensorStorage*       Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                        const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y);

        // -------------------------------------------------------- //
        //  Matrix multiplication                                    //
        // -------------------------------------------------------- //
        static void matmul(const TensorStorage* Xx, const TensorStorage* Xy,
                           TensorStorage* Xz,
                           size_t xN, size_t yN, size_t zN,
                           sys::DeviceType dev);
    private:
        static void dispatch(const f32* x, const f32* y, f32* z,
                             const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                             const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                             const std::vector<i64>& shape_z, const std::vector<i64>& stride_z,
                             BroadcastKind kind, sys::DeviceType dev,
                             char op); // op: '+', '-', '*', '/'
    };
} //namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_MATRIX_HPP