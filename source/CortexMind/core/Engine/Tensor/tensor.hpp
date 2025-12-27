#ifndef CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP
#define CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP

#include <CortexMind/core/Engine/Storage/storage.hpp>
#include <CortexMind/core/Engine/Graph/Node/node.hpp>
#include <CortexMind/core/Engine/AVX/ops.hpp>
#include <CortexMind/core/Engine/AVX/matrix.hpp>
#include <CortexMind/core/Tools/debug.hpp>

#include <memory>
#include <vector>
#include <array>
#include <numeric>
#include <functional>
#include <random>
#include <cstring>
#include <algorithm>
#include <iomanip>

namespace cortex::_fw {
	// @brief MindTensor class representing a multi-dimensional tensor
	class MindTensor {
	public:
		// ----- Constructors and Destructor -----
		MindTensor(bool requires_grad = false);
		explicit MindTensor(std::vector<int64_t> shape, bool requires_grad = false);
		explicit MindTensor(std::initializer_list<int64_t> shape, bool requires_grad = false);
		MindTensor(const std::vector<int64_t>& shape, const float* data, bool requires_grad = false);

		MindTensor(const MindTensor& other) = delete;
		MindTensor& operator=(const MindTensor& other) = delete;
		MindTensor(MindTensor&& other) noexcept = default;
		MindTensor& operator=(MindTensor&& other) noexcept = default;

		~MindTensor() = default;
		
		// ----- Public Methods -----
		
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
			std::vector<int64_t> idx_vec{ static_cast<int64_t>(indices)... };
			int64_t offset = compute_offset(idx_vec);
			return this->m_stor->m_data[offset];
		}

		template<typename... Idx>
		[[nodiscard]] const float& at(Idx... indices) const noexcept {
			std::vector<int64_t> idx_vec{ static_cast<int64_t>(indices)... };
			int64_t offset = compute_offset(idx_vec);
			return this->m_stor->m_data[offset];
		}

		// @brief Get pointer to the underlying data
		[[nodiscard]] float* data() noexcept;
		// @brief Get const pointer to the underlying data
		[[nodiscard]] const float* data() const noexcept;
		// @brief Get the shape of the tensor
		[[nodiscard]] const std::vector<int64_t>& shape() const noexcept;
		// @brief Get the strides of the tensor
		[[nodiscard]] const std::vector<int64_t>& strides() const noexcept;
		// @brief Get the number of elements in the tensor
		[[nodiscard]] size_t numel() const noexcept;
		// @brief Get the size of the underlying storage
		[[nodiscard]] std::size_t size() const noexcept;
		// @brief Check if the tensor is empty
		[[nodiscard]] bool empty() const noexcept;
		// @brief Check if the tensor requires gradient computation
		[[nodiscard]] bool requires_grad() const noexcept;

		// @brief Perform backpropagation to compute gradients
		void backward() noexcept;
		// @brief Print the tensor contents
		void print() const noexcept;
		// @brief Initialize tensor with random values from a uniform distribution
		void uniform_rand(float lower = 0.0f, float upper = 1.0f) noexcept;
		// @brief Initialize tensor with random values from a normal distribution
		void zero() noexcept;
		// @brief Initialize tensor with ones
		void ones() noexcept;
		// @brief Fill tensor with a specific value
		void fill(float value) noexcept;
		// @brief Allocate memory for the tensor
		void allocate() noexcept;
		// @brief Resize the tensor to a new shape
		void resize(const std::vector<int64_t>& new_shape) noexcept;
		// @brief Set whether the tensor requires gradient computation
		void require_grad(bool requires_grad) noexcept;

		//[[nodiscard]] MindTensor contiguous_data() noexcept;
		//[[nodiscard]] MindTensor contiguous_data() const noexcept;

		// @brief Transform N-D tensor to 1-D tensor
		[[nodiscard]] MindTensor flatten() const noexcept;
		// @brief Matrix multiplication with another tensor
		[[nodiscard]] MindTensor matmul(const MindTensor& other) const noexcept;
		// @brief Transpose the tensor
		[[nodiscard]] MindTensor transpose() const noexcept;
		// @brief Permute the dimensions of the tensor
		[[nodiscard]] MindTensor permute(std::vector<int64_t> axes) const noexcept;
		// @brief Slice the tensor along a specific dimension
		[[nodiscard]] MindTensor slice(int64_t dim, int64_t start, int64_t end) const noexcept;
		// @brief Clone the tensor
		[[nodiscard]] MindTensor copy() const noexcept;
		// @brief Compute the sum of all elements or along a specific dimension
		[[nodiscard]] MindTensor sum() const;
		// @brief Compute the sum along a specific dimension
		[[nodiscard]] MindTensor sum(int64_t dim, bool keepdim = false) const;
		// @brief Compute the mean of all elements or along a specific dimension
		[[nodiscrad]] MindTensor mean() const noexcept;
		// @brief Compute the mean along a specific dimension
		[[nodiscard]] MindTensor mean(int64_t dim, bool keepdim = false) const;
		// @brief Reshape the tensor to a new shape
		[[nodiscard]] MindTensor view(const std::vector<int64_t>& new_shape) const;

		MindTensor operator+(const MindTensor& other) const noexcept;
		MindTensor operator-(const MindTensor& other) const noexcept;
		MindTensor operator*(const MindTensor& other) const noexcept;
		MindTensor operator/(const MindTensor& other) const noexcept;

		MindTensor& operator+=(const MindTensor& other) noexcept;
		MindTensor& operator-=(const MindTensor& other) noexcept;
		MindTensor& operator*=(const MindTensor& other) noexcept;
		MindTensor& operator/=(const MindTensor& other) noexcept;

		MindTensor operator+(float scalar) const noexcept;
		MindTensor operator-(float scalar) const noexcept;
		MindTensor operator*(float scalar) const noexcept;
		MindTensor operator/(float scalar) const noexcept;

		MindTensor& operator+=(float scalar) noexcept;
		MindTensor& operator-=(float scalar) noexcept;
		MindTensor& operator*=(float scalar) noexcept;
		MindTensor& operator/=(float scalar) noexcept;
	private:
		std::shared_ptr<TensorStorage> m_stor;
		std::shared_ptr<meta::AutogradMeta> m_meta;
		std::vector<int64_t> m_shape;
		std::vector<int64_t> m_strides;
		int64_t m_offset = 0;

		[[nodiscard]] static std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape) noexcept;
		[[nodiscard]] int64_t compute_offset(const std::vector<int64_t>& indices) const noexcept;
		[[nodiscard]] bool is_contiguous() const noexcept;
	};

	inline MindTensor operator+(float scalar, const MindTensor& tensor) {
		return tensor + scalar;
	}

	inline MindTensor operator*(float scalar, const MindTensor& tensor) {
		return tensor * scalar;
	}

} // namespace cortex::_fw

#endif // CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP

/*
Tensor shouldn't return copy, ixpection of copy(), because of speed but this also is against auto-grad. Tensor should support N-D slicing and indexing.
Because of all layers, optimizers, loss functions depends on Tensor class, and all of them are not using the one D and that creating index issues.
*/