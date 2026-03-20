//
// Created by muham on 16.03.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_TENSOR_UTILS_HPP
#define CORTEXMIND_CORE_TOOLS_TENSOR_UTILS_HPP

#include <CortexMind/core/Tools/err.hpp>
#include <CortexMind/core/Tools/params.hpp>
#include <algorithm>
#include <vector>
#include <utility>

namespace cortex::_fw {
    /**
     * @brief   Computes linear offset from multidimensional indices
     * @param   indices   Multi-dimensional index vector (e.g. {2, 3, 1})
     * @param   stride    Stride vector matching indices size
     * @return  Linear 1D offset into contiguous buffer
     *
     * @pre     indices.size() == stride.size()
     * @note    offset = ∑ (indices[i] × stride[i])
     */
    [[nodiscard]]
    inline i64 compute_offset(const std::vector<i64>& indices, const std::vector<i64>& stride) {
        CXM_ASSERT(indices.size() == stride.size(), "cortex::_fw::compute_offset()", "Index size and stride size must be same.");

        i64 output = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            output += indices[i] * stride[i];
        }
        return output;
    }
    /**
     * @brief   Computes row-major strides from shape
     * @param   shape   Tensor dimensions (e.g. {N, C, H, W})
     * @return  Stride vector where stride[i] = product of shape[i+1..end]
     *
     * @pre     shape is non-empty
     * @note    Last stride is always 1
     * @note    Example: shape {2,3,4} → strides {12, 4, 1}
     */
    [[nodiscard]]
    inline std::vector<i64> compute_stride(const std::vector<i64>& shape) {
        CXM_ASSERT(!shape.empty(), "cortex::_fw::compute_stride()", "Shape cannot be empty.");

        const size_t ndim = shape.size();
        std::vector<i64> output(ndim);

        output[ndim - 1] = 1;
        for (i64 i = static_cast<i64>(ndim) - 2; i >= 0; --i) {
            output[i] = output[i + 1] * shape[i + 1];
        }

        return output;
    }
    /**
     * @brief   Computes total number of elements from shape
     * @param   shape   Tensor dimensions
     * @return  Product of all dimensions (numel)
     *
     * @pre     shape is non-empty
     * @note    Warns if any dimension ≤ 0
     * @note    Returns 0 if any dimension ≤ 0 (invalid shape)
     */
    [[nodiscard]]
    inline i64 compute_numel(const std::vector<i64>& shape) {
        CXM_ASSERT(!shape.empty(), "cortex::_fw::compute_numel()", "Shape cannot be empty.");

        i64 output = 1;
        for (const i64 item : shape) {
            CXM_WARN(item > 0, "cortex::_fw::compute_numel()", "Shape contains zero or negative dimension.");
            output *= item;
        }
        return output;
    }
    /**
     * @brief   Checks if shape is valid (non-empty and all dims > 0)
     * @param   shape   Tensor dimensions
     * @return  true if valid, false otherwise
     */
    [[nodiscard]]
    inline bool is_valid_shape(const std::vector<i64>& shape) {
        return !shape.empty() && std::ranges::all_of(shape, [](const i64 dim) { return dim > 0; });
    }
    /**
     * @brief   Checks if memory layout is contiguous (row-major)
     * @param   shape    Tensor dimensions
     * @param   strides  Stride vector
     * @return  true if strides match computed row-major strides
     *
     * @pre     shape.size() == strides.size()
     */
    [[nodiscard]]
    inline bool is_contiguous(const std::vector<i64>& shape, const std::vector<i64>& strides) {
        CXM_ASSERT(shape.size() == strides.size(), "cortex::_fw::is_contiguous()", "Shape and strides must have the same size.");

        const std::vector<i64> expected = compute_stride(shape);
        return strides == expected;
    }
    /**
     * @brief   Checks if two shapes are broadcastable
     * @param   a   First shape
     * @param   b   Second shape
     * @return  true if broadcastable (NumPy/PyTorch rules)
     *
     * @note    Right-aligned comparison: dimensions of 1 are stretchable
     * @note    If incompatible → returns false
     */
    [[nodiscard]]
    inline bool is_broadcastable(const std::vector<i64>& a, const std::vector<i64>& b) {
        const size_t na = a.size();
        const size_t nb = b.size();
        const size_t n  = std::max(na, nb);

        for (size_t i = 0; i < n; ++i) {
            const i64 da = i < na ? a[na - 1 - i] : 1;
            const i64 db = i < nb ? b[nb - 1 - i] : 1;
            if (da != db && da != 1 && db != 1) return false;
        }
        return true;
    }
    /**
     * @brief   Computes broadcasted shape (union of two shapes)
     * @param   a   First shape
     * @param   b   Second shape
     * @return  Broadcasted shape (max dims, expanded 1s)
     *
     * @pre     Shapes must be broadcastable (checked via is_broadcastable)
     */
    [[nodiscard]]
    inline std::vector<i64> broadcast_shape(const std::vector<i64>& a, const std::vector<i64>& b) {
        CXM_ASSERT(is_broadcastable(a, b), "cortex::_fw::broadcast_shape()", "Shapes are not broadcastable.");

        const size_t na = a.size();
        const size_t nb = b.size();
        const size_t n  = std::max(na, nb);

        std::vector<i64> output(n);
        for (size_t i = 0; i < n; ++i) {
            const i64 da = i < na ? a[na - 1 - i] : 1;
            const i64 db = i < nb ? b[nb - 1 - i] : 1;
            output[n - 1 - i] = std::max(da, db);
        }
        return output;
    }

    static std::vector<i64> pad_shape(const std::vector<i64>& shape, size_t ndim) {
        std::vector<i64> out(ndim, 1);
        const size_t offset = ndim - shape.size();
        for (size_t i = 0; i < shape.size(); ++i)
            out[offset + i] = shape[i];
        return out;
    }

    static std::vector<i64> pad_stride(const std::vector<i64>& stride, const std::vector<i64>& shape, size_t ndim) {
        std::vector<i64> padded_shape = pad_shape(shape, ndim);
        std::vector<i64> out(ndim, 0);
        const size_t offset = ndim - stride.size();
        for (size_t i = 0; i < stride.size(); ++i)
            out[offset + i] = padded_shape[offset + i] == 1 ? 0 : stride[i];
        return out;
    }

    template<typename Op>
    void broadcast_cpu(
    const f32* __restrict__ Xx,
    const f32* __restrict__ Xy,
    f32* __restrict__ Xz,
    const std::vector<i64>& shape_x,
    const std::vector<i64>& stride_x,
    const std::vector<i64>& shape_y,
    const std::vector<i64>& stride_y,
    const std::vector<i64>& shape_out,
    Op op)
    {
        const size_t numel = static_cast<size_t>(compute_numel(shape_out));
        const i64 ndim = static_cast<i64>(shape_out.size());

        for (size_t out_idx = 0; out_idx < numel; ++out_idx) {
            size_t remaining = out_idx;
            i64 ix = 0, iy = 0;
            for (i64 d = ndim - 1; d >= 0; --d) {
                const i64 coord = static_cast<i64>(remaining) % shape_out[d];
                remaining /= static_cast<size_t>(shape_out[d]);
                ix += (shape_x[d] == 1 ? 0 : coord) * stride_x[d];
                iy += (shape_y[d] == 1 ? 0 : coord) * stride_y[d];
            }
            Xz[out_idx] = op(Xx[ix], Xy[iy]);
        }
    }
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_TOOLS_TENSOR_UTILS_HPP