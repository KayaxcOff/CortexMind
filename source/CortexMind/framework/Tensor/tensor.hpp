//
// Created by muham on 18.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP
#define CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP

#include <CortexMind/framework/Memory/device.hpp>
#include <CortexMind/framework/Gradient/flow.hpp>
#include <CortexMind/framework/Tools/params.hpp>
#include <CortexMind/framework/Storage/stor.hpp>
#include <initializer_list>
#include <memory>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief Core tensor class representing multidimensional arrays with device support.
     *
     * `MindTensor` is the central data structure for neural network computations.
     * It supports both CPU and GPU (CUDA) backends, automatic differentiation,
     * and various mathematical operations.
     */
    class MindTensor {
    public:
        MindTensor();
        /**
         * @brief Constructs a tensor with given shape on specified device.
         * @param shape         Tensor dimensions
         * @param device        Target device (host or cuda)
         * @param requires_grad Whether to track gradients for this tensor
         */
        explicit MindTensor(const std::vector<i64>& shape, sys::deviceType device = sys::deviceType::host, bool requires_grad = false);
        MindTensor(std::initializer_list<i64> shape, sys::deviceType device = sys::deviceType::host, bool requires_grad = false);
        /**
         * @brief Constructs a tensor from an existing storage (shared ownership).
         */
        explicit MindTensor(const TensorStorage &tensor_storage, bool requires_grad = false);
        /**
         * @brief Constructs a tensor from external data.
         */
        MindTensor(const std::vector<i64>& shape, const f32* data, sys::deviceType device = sys::deviceType::host, bool requires_grad = false);
        MindTensor(const TensorStorage& storage, MindTensor& _grad);
        MindTensor(const MindTensor& other);
        MindTensor(MindTensor&& other) noexcept;
        ~MindTensor();

        /**
         * @brief Returns a mutable pointer to the underlying data.
         * @return Pointer to the first element (CPU or GPU memory)
         */
        [[nodiscard]]
        f32* get();
        /**
         * @brief Returns a const pointer to the underlying data.
         * @return Const pointer to the first element
         */
        [[nodiscard]]
        const f32* get() const;
        /**
         * @brief Returns the shape of the tensor.
         */
        [[nodiscard]]
        const std::vector<i64>& shape() const;
        /**
         * @brief Returns whether gradient tracking is enabled for this tensor.
         */
        [[nodiscard]]
        bool requires_grad() const;
        /**
         * @brief Returns the current device type (host or cuda).
         */
        [[nodiscard]]
        sys::deviceType device() const;

        /**
         * @brief Returns the total number of elements in the tensor.
         */
        [[nodiscard]]
        size_t numel() const;
        /**
         * @brief Returns the number of dimensions of the tensor.
         */
        [[nodiscard]]
        size_t ndim() const;

        /**
         * @brief Computes the mean of all elements in the tensor.
         */
        [[nodiscard]]
        f32 mean() const;
        /**
         * @brief Computes the variance of all elements in the tensor.
         */
        [[nodiscard]]
        f32 variance() const;
        /**
         * @brief Computes the standard deviation of all elements in the tensor.
         */
        [[nodiscard]]
        f32 stdv() const;
        /**
         * @brief Returns the maximum value in the tensor.
         */
        [[nodiscard]]
        f32 max() const;
        /**
         * @brief Returns the minimum value in the tensor.
         */
        [[nodiscard]]
        f32 min() const;

        /**
         * @brief Fills the tensor with 1.0f (in-place).
         */
        void ones() const;
        /**
         * @brief Fills the tensor with 0.0f (in-place).
         */
        void zero() const;
        /**
         * @brief Fills the tensor with a constant value (in-place).
         * @param value Value to fill
         */
        void fill(f32 value) const;
        /**
         * @brief Fills the tensor with random values from uniform distribution [min, max].
         * @param min Lower bound (default 0.0)
         * @param max Upper bound (default 1.0)
         */
        void rand(f32 min = 0.0f, f32 max = 1.0f) const;
        /**
         * @brief Performs backward pass starting from this tensor (assumes scalar loss).
         */
        void backward() const;
        /**
         * @brief Performs backward pass with a given gradient tensor.
         * @param other Gradient to backpropagate
         */
        void backward(MindTensor& other) const;

        [[nodiscard]]
        MindTensor dot(MindTensor other);
        [[nodiscard]]
        MindTensor pow(f32 exp = 2);
        [[nodiscard]]
        MindTensor sqrt();
        [[nodiscard]]
        MindTensor log();
        [[nodiscard]]
        MindTensor exp();
        [[nodiscard]]
        MindTensor transpose() const;
        [[nodiscard]]
        MindTensor sum() const;

        /**
         * @brief Returns a reference to the gradient tensor (mutable).
         */
        [[nodiscard]]
        MindTensor& grad();
        /**
         * @brief Returns a const reference to the gradient tensor.
         */
        [[nodiscard]]
        const MindTensor& grad() const;

        MindTensor operator+(const MindTensor& other) const;
        MindTensor operator-(const MindTensor& other) const;
        MindTensor operator*(const MindTensor& other) const;
        MindTensor operator/(const MindTensor& other) const;

        MindTensor operator+=(const MindTensor& other);
        MindTensor operator-=(const MindTensor& other);
        MindTensor operator*=(const MindTensor& other);
        MindTensor operator/=(const MindTensor& other);

        MindTensor operator+(f32 value) const;
        MindTensor operator-(f32 value) const;
        MindTensor operator*(f32 value) const;
        MindTensor operator/(f32 value) const;

        MindTensor operator+=(f32 value);
        MindTensor operator-=(f32 value);
        MindTensor operator*=(f32 value);
        MindTensor operator/=(f32 value);

        MindTensor& operator=(const MindTensor& other);
        MindTensor& operator=(MindTensor&& other) noexcept;

        friend std::ostream& operator<<(std::ostream& os, const MindTensor& tensor);
    private:
        std::shared_ptr<meta::GradientFlow> flow_;
        std::shared_ptr<TensorStorage> storage_;
        std::unique_ptr<MindTensor> gradient_;

        bool m_grad_flag;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP