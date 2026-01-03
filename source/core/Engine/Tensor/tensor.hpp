//
// Created by muham on 29.12.2025.
//

#ifndef CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP
#define CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP

#include <core/Engine/Storage/storage.hpp>
#include <core/Graph/meta.hpp>

#include <unordered_set>

namespace cortex::_fw {
	/// @brief Core tensor class that supports multidimensional arrays with autograd support.
	///
	/// @details
	/// MindTensor represents an N-dimensional tensor with contiguous memory storage.
	/// It supports element-wise access, arithmetic operations, slicing, reshaping,
	/// and autograd (gradient computation). The class uses shared_ptr for memory
	/// management and can optionally track gradients for deep learning applications.
    class MindTensor {
    public:
    	/// @brief Constructs an empty tensor with optional gradient tracking.
        explicit MindTensor(bool requires_grad = false);

    	/// @brief Constructs a tensor with a given shape.
    	/// @param shape Dimensions of the tensor
    	/// @param requires_grad Whether gradients should be tracked
        explicit MindTensor(const std::vector<int64_t> &shape, bool requires_grad = false);

    	/// @brief Constructs a tensor with an initializer list of dimensions.
        explicit MindTensor(std::initializer_list<int64_t> shape, bool requires_grad = false);

    	/// @brief Constructs a tensor with given shape and copies data from a raw pointer.
        MindTensor(const std::vector<int64_t>& shape, const float* data, bool requires_grad = false);

        // @brief Access element at given indices
		// @note Indices are provided as a vector of int64_t
		// @return Reference to the element at the specified indices
		[[nodiscard]] float& at(const std::vector<int64_t>& indices) noexcept;
		// @brief Access element at given indices (const version)
		// @note Indices are provided as a vector of int64_t
		// @return Const reference to the element at the specified indices
		[[nodiscard]] const float& at(const std::vector<int64_t>& indices) const noexcept;

		template<typename... Idx>
		[[nodiscard]] float& at(Idx... indices) noexcept {
			const std::vector<int64_t> idxVec{ static_cast<int64_t>(indices)... };
			const int64_t offset = compute_offset(idxVec);
			return this->m_stor->ptr()[offset];
		}

		template<typename... Idx>
		[[nodiscard]] const float& at(Idx... indices) const noexcept {
			const std::vector<int64_t> idxVec{ static_cast<int64_t>(indices)... };
			const int64_t offset = compute_offset(idxVec);
			return this->m_stor->ptr()[offset];
		}

    	[[nodiscard]] float* data() noexcept;                        ///< Returns pointer to underlying storage
    	[[nodiscard]] const float* data() const noexcept;            ///< Const version of data()
    	[[nodiscard]] const std::vector<int64_t>& shape() const noexcept;    ///< Returns tensor shape
    	[[nodiscard]] const std::vector<int64_t>& strides() const noexcept;  ///< Returns tensor strides
    	[[nodiscard]] size_t total() const noexcept;                 ///< Total number of elements
    	[[nodiscard]] std::size_t size() const noexcept;             ///< Size of the underlying storage
    	[[nodiscard]] bool empty() const noexcept;                   ///< Check if tensor has no elements
    	[[nodiscard]] bool requires_grad() const noexcept;           ///< Check if gradient tracking is enabled
    	[[nodiscard]] float mean() const;                   ///< Compute mean of all elements

    	void backward() const;                                     ///< Compute gradients via backpropagation
    	void print() const noexcept;                                  ///< Print tensor contents to stdout
    	/**
		 * @brief Fills the tensor with random values drawn from a uniform distribution.
		 *
		 * This function populates all elements of the tensor with random floating-point numbers
		 * sampled uniformly from the range [lower, upper). By default, the range is [0.0, 1.0),
		 * meaning values are distributed evenly between 0 (inclusive) and 1 (exclusive).
		 *
		 * @param lower The lower bound of the uniform distribution (inclusive). Default is 0.0f.
		 * @param upper The upper bound of the uniform distribution (exclusive). Default is 1.0f.
		 *
		 * @note The function is marked `noexcept`, guaranteeing that it does not throw exceptions.
		 * @note Suitable for initializing weights, biases, or any tensor requiring random values.
		 */
    	void uniform_rand(float lower = 0.0f, float upper = 1.0f) const noexcept;
    	void zero() const noexcept;                                   ///< Fill tensor with zeros
    	void ones() const noexcept;                                   ///< Fill tensor with ones
    	void fill(float value) const noexcept;                        ///< Fill tensor with a specific value
    	/**
		 * @brief Allocates memory for the tensor's underlying storage.
		 *
		 * This function initializes the internal `TensorStorage` structure, which holds
		 * the actual data of the tensor. After calling this function, the tensor will
		 * have a valid memory buffer ready for storing values. No values are set during
		 * allocation; they remain uninitialized.
		 *
		 * @note The function is marked `noexcept`, ensuring it does not throw exceptions.
		 * @note It must be called before performing any operations that require
		 *       accessing or modifying the tensor's data.
		 */
    	void allocate() noexcept;
    	/**
		 * @brief Resizes the tensor to a new shape and allocates storage accordingly.
		 *
		 * This function updates the tensor's shape and recalculates the corresponding
		 * strides to ensure proper memory access. It also allocates a new `TensorStorage`
		 * instance large enough to hold all elements for the new shape.
		 *
		 * @param new_shape A vector representing the desired dimensions of the tensor.
		 *                  Each element specifies the size along that axis.
		 *
		 * @note The total number of elements is computed as the product of all dimensions
		 *       in `new_shape`.
		 * @note Existing data is discarded; the tensor's memory is reallocated to fit
		 *       the new size.
		 * @note The internal offset (`m_offset`) is reset to 0.
		 * @note This function is `noexcept` and guaranteed not to throw exceptions.
		 *
		 * @example
		 * // Resize a tensor to shape [2, 3, 4]
		 * MindTensor tensor;
		 * x.resize({2, 3, 4});
		 */
    	void resize(const std::vector<int64_t>& new_shape) noexcept;  ///< Resize tensor to new shape
    	void require_grad(bool requires_grad) const noexcept;               ///< Enable/disable gradient tracking

    	/**
		 * @brief Returns a flattened version of the tensor, preserving the batch dimension.
		 *
		 * This function collapses all dimensions of the tensor except the first one (batch)
		 * into a single dimension, producing a 2D tensor with shape `[batch, features]`, where
		 * `features` is the product of all non-batch dimensions.
		 *
		 * The returned tensor shares the same underlying storage (`TensorStorage`) and offset
		 * as the original tensor. No data is copied; only the shape and strides are updated
		 * to reflect the flattened view.
		 *
		 * @return MindTensor A new tensor with shape `[batch, features]` that shares storage
		 *                    with the original tensor.
		 *
		 * @note This operation is `noexcept` and does not throw exceptions.
		 * @note The `requires_grad` property is preserved in the new tensor.
		 *
		 * @example
		 * // Flatten a tensor of shape [10, 3, 32, 32] into [10, 3072]
		 * tensor input({10, 3, 32, 32});
		 * tensor flat = input.flatten();
		 */
    	[[nodiscard]] MindTensor flatten() const noexcept;
    	/**
		 * @brief Performs 2D matrix multiplication between this tensor and another tensor.
		 *
		 * Both tensors must be 2-dimensional. This function computes the matrix product
		 * `C = A * B`, where `A` is this tensor and `B` is the `other` tensor. The resulting
		 * tensor `C` has shape `[A.rows, B.cols]`.
		 *
		 * @param other The tensor to multiply with. Must have a compatible shape
		 *              such that the number of columns of this tensor equals
		 *              the number of rows of `other`.
		 * @return MindTensor A new tensor containing the result of the matrix multiplication.
		 *                    The `requires_grad` property is preserved from the original tensor.
		 *
		 * @throws Error If either tensor is not 2D or if the inner dimensions do not match.
		 *
		 * @note The function uses optimized AVX2 instructions for efficient computation.
		 * @note The output tensor is newly allocated; data is not shared with the inputs.
		 *
		 * @example
		 * // Multiply a [2, 3] tensor with a [3, 4] tensor to get a [2, 4] result
		 * tensor A({2, 3});
		 * tensor B({3, 4});
		 * tensor C = A.matmul(B);
		 */
    	[[nodiscard]] MindTensor matmul(const MindTensor& other) const;
    	/**
		 * @brief Returns the transpose of a 2D tensor.
		 *
		 * This function swaps the two dimensions of a 2D tensor, producing a new tensor
		 * where rows become columns and columns become rows. The resulting tensor shares
		 * the same underlying storage (`TensorStorage`) and offset as the original tensor,
		 * so no data is copied; only the strides and shape are adjusted to reflect the transposed view.
		 *
		 * @return MindTensor A new tensor with transposed shape `[cols, rows]`.
		 *                    The `requires_grad` property is preserved from the original tensor.
		 *
		 * @throws Error If the tensor is not 2-dimensional.
		 *
		 * @note This operation is `noexcept` in terms of memory allocation since no new storage is created.
		 * @note Because storage is shared, modifying elements in the transposed tensor will affect the original tensor.
		 *
		 * @example
		 * // Transpose a [2, 3] tensor to get a [3, 2] tensor
		 * tensor input({2, 3});
		 * tensor transposed = input.transpose();
		 */
    	[[nodiscard]] MindTensor transpose() const;
    	/**
		 * @brief Returns a view of the tensor with permuted dimensions according to the specified axes.
		 *
		 * This function rearranges the dimensions of the tensor based on the order provided
		 * in the `axes` vector. The resulting tensor shares the same underlying storage
		 * (`TensorStorage`) and offset as the original tensor, so no data is copied; only
		 * the shape and strides are updated to reflect the new dimension order.
		 *
		 * @param axes A vector specifying the desired permutation of the tensor's dimensions.
		 *             Each element must be a valid axis index, and the size of `axes` must
		 *             match the number of dimensions of the tensor.
		 *
		 * @return MindTensor A new tensor with permuted dimensions that shares storage
		 *                    with the original tensor. The `requires_grad` property is preserved.
		 *
		 * @throws Error If the number of axes does not match the tensor's rank, or if
		 *               any axis index is invalid.
		 *
		 * @note This operation is `noexcept` in terms of memory allocation since no new storage is created.
		 * @note Modifying elements in the permuted tensor will affect the original tensor
		 *       because storage is shared.
		 *
		 * @example
		 * // Permute a [2, 3, 4] tensor to [4, 2, 3]
		 * tensor input({2, 3, 4});
		 * tensor permuted = input.permute({2, 0, 1});
		 */
    	[[nodiscard]] MindTensor permute(const std::vector<int64_t> &axes) const;
    	[[nodiscard]] MindTensor slice(int64_t dim, int64_t start, int64_t end) const; ///< Slice along a dimension
    	[[nodiscard]] MindTensor copy() const;               ///< Deep copy of tensor
    	[[nodiscard]] MindTensor sum() const;                         ///< Sum all elements
    	[[nodiscard]] MindTensor sum(int64_t dim, bool keep = false) const;  ///< Sum along a specific dimension
    	[[nodiscard]] MindTensor mean(int64_t dim, bool keep = false) const; ///< Mean along a specific dimension
    	[[nodiscard]] MindTensor view(const std::vector<int64_t>& new_shape) const; ///< Reshape tensor
    	[[nodiscard]] MindTensor sqrt() const;				  ///< Tensor's sqrt
    	[[nodiscard]] MindTensor unsqueeze(int64_t dim) const; ///< Unsqueeze tensor
    	[[nodiscard]] MindTensor squeeze(int64_t dim) const; ///< Squeeze tensor
    	[[nodiscard]] MindTensor reshape(const std::vector<int64_t>& new_shape) const;
    	/**
		 * @brief Repeat tensor elements along each dimension by explicit data duplication.
		 *
		 * @details
		 * This operation creates a new tensor with expanded shape where elements are
		 * physically copied (unlike reshape/view which only change metadata).
		 *
		 * Each value in @p repeats specifies how many times the tensor is repeated
		 * along the corresponding dimension.
		 *
		 * @warning
		 * This function multiplies the size of each dimension by the given repeat value.
		 * It does NOT perform broadcasting.
		 *
		 * Common pitfall (Dense bias example):
		 *   bias shape = (1, F)
		 *   ❌ bias.repeat({batch, F})  -> produces shape (batch, F * F)  [WRONG]
		 *   ✅ bias.repeat({batch, 1})  -> produces shape (batch, F)      [CORRECT]
		 *
		 * To repeat across the batch dimension only, always use repeat({batch, 1})
		 * and never repeat the feature dimension.
		 *
		 * @param repeats Number of repetitions per dimension (must match tensor rank)
		 * @return A new tensor with repeated data and newly allocated storage
		 */
    	[[nodiscard]] MindTensor repeat(const std::vector<int64_t>& repeats) const;

    	[[nodiscard]] MindTensor& grad();									 /// < Return tensor gradient
    	[[nodiscard]] const MindTensor& grad() const;                        ///< Return const tensor gradient

		MindTensor operator+(const MindTensor& other) const;
		MindTensor operator-(const MindTensor& other) const;
		MindTensor operator*(const MindTensor& other) const;
		MindTensor operator/(const MindTensor& other) const;

		MindTensor& operator+=(const MindTensor& other);
		MindTensor& operator-=(const MindTensor& other);
		MindTensor& operator*=(const MindTensor& other);
		MindTensor& operator/=(const MindTensor& other);

		MindTensor operator+(float scalar) const;
		MindTensor operator-(float scalar) const;
		MindTensor operator*(float scalar) const;
		MindTensor operator/(float scalar) const;

		MindTensor& operator+=(float scalar);
		MindTensor& operator-=(float scalar);
		MindTensor& operator*=(float scalar);
		MindTensor& operator/=(float scalar);

    private:
        std::shared_ptr<TensorStorage> m_stor;
        std::shared_ptr<meta::AutoDiff<MindTensor>> m_flow;
    	std::shared_ptr<MindTensor> m_grad;
        std::vector<int64_t> m_shape;
        std::vector<int64_t> m_strides;
        int64_t m_offset = 0;

        [[nodiscard]] static std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape) noexcept;
        [[nodiscard]] int64_t compute_offset(const std::vector<int64_t>& indices) const noexcept;
        [[nodiscard]] bool is_contiguous() const noexcept;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP