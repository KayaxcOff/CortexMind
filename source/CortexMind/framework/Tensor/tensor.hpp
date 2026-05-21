//
// Created by muham on 15.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP
#define CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP

#include <CortexMind/framework/Gradient/flow.hpp>
#include <CortexMind/framework/Gradient/pack.hpp>
#include <CortexMind/framework/Storage/stor.hpp>
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/tensor_meta.hpp>
#include <concepts>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief Multidimensional array with automatic differentiation support.
     *
     * `Tensor` is the primary data structure used for all computations in the framework.
     * It manages memory through `TensorStorage`, supports multiple devices (Host/CUDA),
     * and integrates with the autograd system via `GradientFlow`.
     */
    class Tensor {
    public:
        /**
         * @brief Default constructor.
         *
         * Creates an empty tensor with no allocated storage,
         * no gradient tracking, and no computation flow.
         */
        Tensor();
        /**
         * @brief Constructs a tensor with the specified shape.
         *
         * Allocates tensor storage on the given device.
         *
         * @param shape Shape of the tensor.
         * @param _device Target device where the tensor will be allocated.
         * @param _requires_grad Enables automatic gradient tracking if true.
         */
        explicit Tensor(const std::vector<i64>& shape, sys::DeviceType _device = sys::DeviceType::kHOST, bool _requires_grad = false);
        /**
         * @brief Constructs a tensor using an initializer list shape.
         *
         * @param shape Shape of the tensor.
         * @param _device Target device where the tensor will be allocated.
         * @param _requires_grad Enables automatic gradient tracking if true.
         */
        explicit Tensor(std::initializer_list<i64> shape, sys::DeviceType _device = sys::DeviceType::kHOST, bool _requires_grad = false);
        /**
         * @brief Constructs a tensor from raw data.
         *
         * Copies the given data into tensor storage.
         *
         * @param shape Shape of the tensor.
         * @param data Pointer to source data.
         * @param _device Target device.
         * @param _requires_grad Enables automatic gradient tracking if true.
         */
        Tensor(const std::vector<i64>& shape, const f32* data, sys::DeviceType _device = sys::DeviceType::kHOST, bool _requires_grad = false);
        /**
         * @brief Constructs a tensor from a packed gradient structure.
         *
         * @param packed Gradient-packed tensor metadata and storage.
         */
        explicit Tensor(const meta::GradientPacked& packed);
        Tensor(const Tensor& other);
        Tensor(Tensor&& other) noexcept;
        ~Tensor();

        /**
         * @brief Returns a mutable reference to an element.
         *
         * This function is only supported for tensors located on the HOST device.
         *
         * @tparam Args Integral index types.
         * @param args Tensor indices.
         * @return Reference to the selected element.
         */
        template<typename ... Args> requires (std::integral<Args> && ...)
        [[nodiscard]]
        f32& at(Args...args) {
            CXM_ASSERT(this->storage_ == nullptr, "Tensor storage is null");
            CXM_ASSERT(this->storage_->device() != sys::DeviceType::kHOST,
                "at() is only supported on HOST tensors");

            const std::vector<i64> indices = { static_cast<i64>(args)... };

            CXM_ASSERT(indices.size() != this->m_shape.size(),
                "Index dimension mismatch: got " + std::to_string(indices.size()) +
                " expected " + std::to_string(this->m_shape.size()));

            for (size_t d = 0; d < indices.size(); ++d) {
                CXM_ASSERT(indices[d] < 0 || indices[d] >= this->m_shape[d],
                    "Index out of bounds at dim " + std::to_string(d) +
                    ": got " + std::to_string(indices[d]) +
                    " size " + std::to_string(this->m_shape[d]));
            }
            const i64 linear = compute_linear_index(this->m_strides, indices, this->m_offset);
            return this->storage_->data()[linear];
        }

        /**
         * @brief Returns a constant reference to an element.
         *
         * This function is only supported for tensors located on the HOST device.
         *
         * @tparam Args Integral index types.
         * @param args Tensor indices.
         * @return Constant reference to the selected element.
         */
        template<typename ... Args> requires (std::integral<Args> && ...)
        [[nodiscard]]
        const f32& at(Args...args) const {
            CXM_ASSERT(this->storage_ == nullptr, "Tensor storage is null");
            CXM_ASSERT(this->storage_->device() != sys::DeviceType::kHOST,
                "at() is only supported on HOST tensors");

            const std::vector<i64> indices = { static_cast<i64>(args)... };

            CXM_ASSERT(indices.size() != this->m_shape.size(),
                "Index dimension mismatch: got " + std::to_string(indices.size()) +
                " expected " + std::to_string(this->m_shape.size()));

            for (size_t d = 0; d < indices.size(); ++d) {
                CXM_ASSERT(indices[d] < 0 || indices[d] >= this->m_shape[d],
                    "Index out of bounds at dim " + std::to_string(d) +
                    ": got " + std::to_string(indices[d]) +
                    " size " + std::to_string(this->m_shape[d]));
            }
            const i64 linear = compute_linear_index(this->m_strides, indices, this->m_offset);
            return this->storage_->data()[linear];
        }

        /**
         * @brief Returns a pointer to the tensor data.
         *
         * @return Pointer to tensor memory.
         */
        [[nodiscard]]
        f32* get();
        /**
         * @brief Returns a constant pointer to the tensor data.
         *
         * @return Constant pointer to tensor memory.
         */
        [[nodiscard]]
        const f32* get() const;
        /**
         * @brief Returns the tensor shape.
         *
         * @return Tensor shape vector.
         */
        [[nodiscard]]
        const std::vector<i64>& shape() const;
        /**
         * @brief Checks whether the tensor has gradient storage.
         *
         * @return True if gradient storage exists.
         */
        [[nodiscard]]
        bool has_grad() const;
        /**
         * @brief Checks whether the tensor storage is empty.
         *
         * @return True if tensor storage is empty.
         */
        [[nodiscard]]
        bool empty() const;
        /**
         * @brief Checks whether the tensor memory layout is contiguous.
         *
         * @return True if tensor is contiguous.
         */
        [[nodiscard]]
        bool contiguous() const;
        /**
         * @brief Returns the device where the tensor is stored.
         *
         * @return Tensor device type.
         */
        [[nodiscard]]
        sys::DeviceType device() const;

        /**
         * @brief Returns the total number of elements.
         *
         * @return Element count.
         */
        [[nodiscard]]
        size_t len() const;
        /**
         * @brief Returns the number of tensor dimensions.
         *
         * @return Dimension count.
         */
        [[nodiscard]]
        size_t ndim() const;

        /**
         * @brief Computes the arithmetic mean of all tensor elements.
         *
         * @return Mean value.
         */
        [[nodiscard]]
        f32 mean() const;
        /**
         * @brief Computes the variance of all tensor elements.
         *
         * @return Variance value.
         */
        [[nodiscard]]
        f32 variance() const;
        /**
         * @brief Computes the standard deviation of all tensor elements.
         *
         * @return Standard deviation value.
         */
        [[nodiscard]]
        f32 stdv() const;
        /**
         * @brief Returns the maximum element value.
         *
         * @return Maximum value.
         */
        [[nodiscard]]
        f32 max() const;
        /**
         * @brief Returns the minimum element value.
         *
         * @return Minimum value.
         */
        [[nodiscard]]
        f32 min() const;
        /**
         * @brief Computes the sum of all tensor elements.
         *
         * @return Sum of all values.
         */
        [[nodiscard]]
        f32 sum_all() const;
        /**
         * @brief Computes the L1 norm of the tensor.
         *
         * @return L1 norm value.
         */
        [[nodiscard]]
        f32 norm1() const;
        /**
         * @brief Computes the L2 norm of the tensor.
         *
         * @return L2 norm value.
         */
        [[nodiscard]]
        f32 norm2() const;

        /**
         * @brief Fills the tensor with a constant value.
         *
         * @param value Fill value.
         */
        void fill(f32 value);
        /**
         * @brief Fills the tensor with zeros.
         */
        void zero();
        /**
         * @brief Fills the tensor with ones.
         */
        void ones();
        /**
         * @brief Fills the tensor with uniformly distributed random values.
         *
         * @param min Minimum random value.
         * @param max Maximum random value.
         */
        void uniform(f32 min = 0.0f, f32 max = 1.0f) const;
        /**
         * @brief Starts backpropagation using the stored gradient tensor.
         *
         * If the tensor is scalar, its gradient is initialized to 1.
         */
        void backward() const;
        /**
         * @brief This function enables navigation to the next node in the gradient flow.
         * @param _grad The gradient of the next node
         * @warning This function should not be used to initiate auto-gradient generation; its purpose is to move to the next node.
         */
        void backward(const Tensor& _grad) const;
        /**
         * @brief Copies external data into the tensor.
         *
         * @param _data Source data pointer.
         */
        void SetData(const f32* _data);
        /**
         * @brief Assigns a shared gradient tensor.
         *
         * @param _grad Shared gradient tensor.
         */
        void SetGrad(const std::shared_ptr<Tensor> &_grad);
        /**
         * @brief Assigns a gradient tensor copy.
         *
         * @param _grad Gradient tensor.
         */
        void SetGrad(const Tensor& _grad);
        /**
         * @brief Sets the computation graph flow object.
         *
         * @param _flow Gradient flow implementation.
         */
        void SetFlow(const std::shared_ptr<meta::GradientFlow>& _flow);

        /**
         * @brief Transfers the tensor to another device.
         *
         * @param _device Target device.
         * @return Tensor located on the target device.
         */
        [[nodiscard]]
        Tensor to(const sys::DeviceType& _device) const;
        /**
         * @brief Performs matrix multiplication.
         *
         * Both tensors must be 2D and compatible for multiplication.
         *
         * @param other Right-hand tensor.
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor matmul(const Tensor& other) const;
        /**
         * @brief Transposes a 2D tensor.
         *
         * @return Transposed tensor.
         */
        [[nodiscard]]
        Tensor transpose() const;
        /**
         * @brief Permutes tensor dimensions.
         *
         * @param dims Dimension permutation order.
         * @return Permuted tensor view.
         * @warning Permuted tensor might be not contiguous and some math functions requires contiguous tensor
         */
        [[nodiscard]]
        Tensor permute(const std::vector<i64>& dims) const;
        /**
         * @brief Reshapes the tensor without copying memory.
         *
         * Tensor must be contiguous.
         *
         * @param _new_shape New tensor shape.
         * @return Reshaped tensor view.
         */
        [[nodiscard]]
        Tensor reshape(const std::vector<i64>& _new_shape) const;
        /**
         * @brief Applies natural logarithm element-wise.
         *
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor log() const;
        /**
         * @brief Applies exponential function element-wise.
         *
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor exp() const;
        /**
         * @brief Raises tensor elements to a power.
         *
         * @param exp Exponent value.
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor pow(f32 exp = 2.0f) const;
        /**
         * @brief Applies square root element-wise.
         *
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor sqrt() const;
        /**
         * @brief Applies reciprocal square root element-wise.
         *
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor rsqrt() const;
        /**
         * @brief Applies sine function element-wise.
         *
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor sin() const;
        /**
         * @brief Applies cosine function element-wise.
         *
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor cos() const;
        /**
         * @brief Computes absolute value element-wise.
         *
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor abs() const;
        /**
         * @brief Returns a sliced tensor view.
         *
         * @param dim Target dimension.
         * @param start Slice start index.
         * @param end Slice end index.
         * @return Sliced tensor view.
         */
        [[nodiscard]]
        Tensor slice(i64 dim, i64 start, i64 end) const;
        /**
         * @brief Computes the sum of all tensor elements.
         *
         * Returns the result as a scalar tensor.
         *
         * @return Scalar tensor containing the sum.
         */
        [[nodiscard]]
        Tensor sum() const;
        [[nodiscard]]
        Tensor sum(const std::vector<i64>& dims, bool keep = true) const;
        /**
         * @brief Negates all tensor elements.
         *
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor neg() const;
        /**
         * @brief Computes the sign of each tensor element.
         *
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor sign() const;
        /**
         * @brief Removes a singleton dimension.
         *
         * @param dim Dimension to remove.
         * @return Tensor view with reduced dimensions.
         */
        [[nodiscard]]
        Tensor squeeze(i64 dim = -1) const;
        /**
         * @brief Inserts a singleton dimension.
         *
         * @param dim Insertion dimension.
         * @return Expanded tensor view.
         */
        [[nodiscard]]
        Tensor unsqueeze(i64 dim) const;
        /**
         * @brief Performs element-wise tensor addition.
         *
         * Supports broadcasting.
         *
         * @param other Right-hand tensor.
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor add(const Tensor& other) const;
        /**
         * @brief Performs element-wise tensor subtraction.
         *
         * Supports broadcasting.
         *
         * @param other Right-hand tensor.
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor sub(const Tensor& other) const;
        /**
         * @brief Performs element-wise tensor multiplication.
         *
         * Supports broadcasting.
         *
         * @param other Right-hand tensor.
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor mul(const Tensor& other) const;
        /**
         * @brief Performs element-wise tensor division.
         *
         * Supports broadcasting.
         *
         * @param other Right-hand tensor.
         * @return Result tensor.
         */
        [[nodiscard]]
        Tensor div(const Tensor& other) const;
        /**
         * @brief addition between tensor and scaler
         * @param value scaler value
         * @return Result tensor
         */
        [[nodiscard]]
        Tensor add(f32 value) const;
        /**
         * @brief subtract between tensor and scaler
         * @param value scaler value
         * @return Result tensor
         */
        [[nodiscard]]
        Tensor sub(f32 value) const;
        /**
         * @brief multiply between tensor and scaler
         * @param value scaler value
         * @return Result tensor
         */
        [[nodiscard]]
        Tensor mul(f32 value) const;
        /**
         * @brief division between tensor and scaler
         * @param value scaler value
         * @return Result tensor
         */
        [[nodiscard]]
        Tensor div(f32 value) const;

        /**
         * @brief Creates a deep copy of the tensor.
         *
         * @return Cloned tensor.
         */
        [[nodiscard]]
        Tensor clone() const;

        /**
         * @brief Returns the gradient tensor.
         *
         * @return Mutable gradient tensor reference.
         */
        [[nodiscard]]
        Tensor& grad();
        /**
         * @brief Returns the gradient tensor.
         *
         * @return Constant gradient tensor reference.
         */
        [[nodiscard]]
        const Tensor& grad() const;

        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const;
        Tensor operator/(const Tensor& other) const;

        Tensor& operator+=(const Tensor& other);
        Tensor& operator-=(const Tensor& other);
        Tensor& operator*=(const Tensor& other);
        Tensor& operator/=(const Tensor& other);

        Tensor operator+(f32 value) const;
        Tensor operator-(f32 value) const;
        Tensor operator*(f32 value) const;
        Tensor operator/(f32 value) const;

        Tensor& operator+=(f32 value);
        Tensor& operator-=(f32 value);
        Tensor& operator*=(f32 value);
        Tensor& operator/=(f32 value);

        bool operator==(const Tensor& other) const;
        bool operator!=(const Tensor& other) const;

        Tensor operator>(const Tensor& other) const;
        Tensor operator<(const Tensor& other) const;
        Tensor operator>=(const Tensor& other) const;
        Tensor operator<=(const Tensor& other) const;

        Tensor& operator=(const Tensor& other);
        Tensor& operator=(Tensor&& other) noexcept;

        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
        friend Tensor operator+(f32 value, const Tensor& tensor2);
        friend Tensor operator-(f32 value, const Tensor& tensor);
        friend Tensor operator*(f32 value, const Tensor& tensor);
        friend Tensor operator/(f32 value, const Tensor& tensor);

    private:
        std::shared_ptr<TensorStorage> storage_;
        std::shared_ptr<Tensor> gradient_;
        std::shared_ptr<meta::GradientFlow> flow_;

        std::vector<i64> m_shape;
        std::vector<i64> m_strides;
        i64 m_offset;

        bool m_requires_grad;

        Tensor(const std::vector<i64>& shape, const std::shared_ptr<TensorStorage>& storage, bool _requires_grad);
        Tensor(const std::vector<i64>& shape, const std::vector<i64>& stride, const std::shared_ptr<TensorStorage>& storage, bool _requires_grad);
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP