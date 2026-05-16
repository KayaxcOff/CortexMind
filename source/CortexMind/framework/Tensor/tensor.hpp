//
// Created by muham on 15.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP
#define CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP

#include <CortexMind/framework/Gradient/flow.hpp>
#include <CortexMind/framework/Gradient/saved_tensor.hpp>
#include <CortexMind/framework/Storage/stor.hpp>
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
         * @brief Default constructor - creates an empty tensor.
         */
        Tensor();
        /**
         * @brief Constructs a tensor with given shape.
         *
         * @param shape        Tensor dimensions
         * @param _device      Target device (default: HOST)
         * @param _requires_grad Whether gradient should be tracked
         */
        explicit Tensor(const std::vector<i64>& shape, sys::DeviceType _device = sys::DeviceType::kHOST, bool _requires_grad = false);
        /**
         * @brief Constructs a tensor from initializer list.
         */
        explicit Tensor(std::initializer_list<i64> shape, sys::DeviceType _device = sys::DeviceType::kHOST, bool _requires_grad = false);
        /**
         * @brief Constructs a tensor from existing data.
         */
        Tensor(const std::vector<i64>& shape, const f32* data, sys::DeviceType _device = sys::DeviceType::kHOST, bool _requires_grad = false);
        explicit Tensor(const meta::GradientPacked& packed);
        Tensor(const Tensor& other);
        Tensor(Tensor&& other) noexcept;
        ~Tensor();

        /**
         * @brief Multidimensional indexing (host memory only).
         */
        template<typename ... Args> requires (std::integral<Args> && ...)
        [[nodiscard]]
        f32& at(Args...args);

        /**
         * @brief Multidimensional indexing as const (host memory only).
         */
        template<typename ... Args> requires (std::integral<Args> && ...)
        [[nodiscard]]
        const f32& at(Args...args) const;

        /**
         * @brief Returns raw mutable pointer to the underlying data.
         */
        [[nodiscard]]
        f32* get();
        /**
         * @brief Returns raw const pointer to the underlying data.
         */
        [[nodiscard]]
        const f32* get() const;
        /**
         * @brief Returns the shape of the tensor.
         */
        [[nodiscard]]
        const std::vector<i64>& shape() const;
        /**
         * @brief Returns whether gradient tracking is enabled.
         */
        [[nodiscard]]
        bool has_grad() const;
        /**
         * @brief Returns true if tensor has zero elements.
         */
        [[nodiscard]]
        bool empty() const;
        /**
         * @brief Returns true if tensor is contiguous in memory.
         */
        [[nodiscard]]
        bool contiguous() const;
        /**
         * @brief Returns the current device of the tensor.
         */
        [[nodiscard]]
        sys::DeviceType device() const;

        /**
         * @brief Returns total number of elements.
         */
        [[nodiscard]]
        size_t len() const;
        /**
         * @brief Returns number of dimensions.
         */
        [[nodiscard]]
        size_t ndim() const;

        /**
         * @brief Returns the mean of all elements.
         */
        [[nodiscard]]
        f32 mean() const;
        /**
         * @brief Returns the variance of all elements.
         */
        [[nodiscard]]
        f32 variance() const;
        /**
         * @brief Returns the standard deviation of all elements.
         */
        [[nodiscard]]
        f32 stdv() const;
        /**
         * @brief Returns the maximum value.
         */
        [[nodiscard]]
        f32 max() const;
        /**
         * @brief Returns the minimum value.
         */
        [[nodiscard]]
        f32 min() const;
        /**
         * @brief Returns the sum of all elements.
         */
        [[nodiscard]]
        f32 sum_all() const;
        /**
         * @brief Returns L1 norm (sum of absolute values).
         */
        [[nodiscard]]
        f32 norm1() const;
        /**
         * @brief Returns L2 norm (Euclidean norm).
         */
        [[nodiscard]]
        f32 norm2() const;

        void fill(f32 value);
        void zero();
        void ones();
        void uniform(f32 min = 0.0f, f32 max = 1.0f) const;
        void backward() const;
        void backward(const Tensor& _grad) const;
        void SetData(const f32* _data);
        void SetGrad(const std::shared_ptr<Tensor> &_grad);
        void SetGrad(const Tensor& _grad);
        void SetFlow(const std::shared_ptr<meta::GradientFlow>& _flow);

        [[nodiscard]]
        Tensor to(const sys::DeviceType& _device) const;
        [[nodiscard]]
        Tensor matmul(const Tensor& other) const;
        [[nodiscard]]
        Tensor transpose() const;
        [[nodiscard]]
        Tensor permute(const std::vector<i64>& dims) const;
        [[nodiscard]]
        Tensor reshape(const std::vector<i64>& _new_shape) const;
        [[nodiscard]]
        Tensor log() const;
        [[nodiscard]]
        Tensor exp() const;
        [[nodiscard]]
        Tensor pow(f32 exp = 2.0f) const;
        [[nodiscard]]
        Tensor sqrt() const;
        [[nodiscard]]
        Tensor rsqrt() const;
        [[nodiscard]]
        Tensor sin() const;
        [[nodiscard]]
        Tensor cos() const;
        [[nodiscard]]
        Tensor abs() const;
        [[nodiscard]]
        Tensor slice(i64 dim, i64 start, i64 end) const;
        [[nodiscard]]
        Tensor sum() const;
        [[nodiscard]]
        Tensor neg() const;
        [[nodiscard]]
        Tensor sign() const;
        [[nodiscard]]
        Tensor squeeze(i64 dim = -1) const;
        [[nodiscard]]
        Tensor unsqueeze(i64 dim) const;
        [[nodiscard]]
        Tensor addition(const Tensor& other) const;
        [[nodiscard]]
        Tensor subtract(const Tensor& other) const;
        [[nodiscard]]
        Tensor multiply(const Tensor& other) const;
        [[nodiscard]]
        Tensor divide(const Tensor& other) const;

        [[nodiscard]]
        Tensor clone() const;

        [[nodiscard]]
        Tensor& grad();
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

        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
        friend Tensor operator-(f32 value, const Tensor& tensor);
        friend Tensor operator*(f32 value, const Tensor& tensor);

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