//
// Created by muham on 15.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP
#define CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP

#include <CortexMind/framework/Gradient/flow.hpp>
#include <CortexMind/framework/Gradient/pack.hpp>
#include <CortexMind/framework/Shape/shape.hpp>
#include <CortexMind/framework/Storage/stor.hpp>
#include <memory>
#include <span>

namespace cortex::_fw {
    class Tensor {
    public:
        Tensor();
        Tensor(const std::initializer_list<i64>& _shape, sys::DeviceType _device = sys::DeviceType::kHOST, bool _requires_grad = false);
        explicit Tensor(const meta::GradientPacked& packed);
        Tensor(const Tensor& other);
        Tensor(Tensor&& other) noexcept;
        ~Tensor();

        template<typename ... Args> requires (std::integral<Args> && ...)
        [[nodiscard]]
        f32& at(Args...args);
        template<typename ... Args> requires (std::integral<Args> && ...)
        [[nodiscard]]
        const f32& at(Args...args) const;

        [[nodiscard]]
        f32* get();
        [[nodiscard]]
        const f32* get() const;
        [[nodiscard]]
        std::span<const i64> shape() const;
        [[nodiscard]]
        bool has_grad() const;
        [[nodiscard]]
        bool is_require() const;
        [[nodiscard]]
        bool empty() const;
        [[nodiscard]]
        bool is_contiguous() const;
        [[nodiscard]]
        sys::DeviceType device() const;

        [[nodiscard]]
        size_t len() const;
        [[nodiscard]]
        size_t ndim() const;

        void fill(f32 value);
        void zero();
        void ones();
        void randn();
        void require_grad();
        void uniform(f32 min = 0.0f, f32 max = 1.0f) const;
        void backward() const;
        void backward(const Tensor& _grad) const;
        void SetData(const f32* _data);
        void SetGrad(const std::shared_ptr<Tensor> &_grad);
        void SetGrad(const Tensor& _grad);
        void SetFlow(const std::shared_ptr<meta::GradientFlow>& _flow);

        [[nodiscard]]
        Tensor mean() const;
        [[nodiscard]]
        Tensor mean(i64 dim, bool keep_dim = true) const;
        [[nodiscard]]
        Tensor variance() const;
        [[nodiscard]]
        Tensor variance(i64 dim, bool keep_dim = true) const;
        [[nodiscard]]
        Tensor stdv() const;
        [[nodiscard]]
        Tensor stdv(i64 dim, bool keep_dim = true) const;
        [[nodiscard]]
        Tensor norm1() const;
        [[nodiscard]]
        Tensor norm1(i64 dim, bool keep_dim = true) const;
        [[nodiscard]]
        Tensor norm2() const;
        [[nodiscard]]
        Tensor norm2(i64 dim, bool keep_dim = true) const;
        [[nodiscard]]
        Tensor max() const;
        [[nodiscard]]
        Tensor max(i64 dim, bool keep_dim = true) const;
        [[nodiscard]]
        Tensor min() const;
        [[nodiscard]]
        Tensor min(i64 dim, bool keep_dim = true) const;
        [[nodiscard]]
        Tensor sum() const;
        [[nodiscard]]
        Tensor sum(i64 dim, bool keep_dim = true) const;
        [[nodiscard]]
        Tensor argmax() const;
        [[nodiscard]]
        Tensor argmax(i64 dim, bool keep_dim = true) const;
        [[nodiscard]]
        Tensor argmin();
        [[nodiscard]]
        Tensor argmin(i64 dim, bool keep_dim = true) const;
        [[nodiscard]]
        Tensor matmul(const Tensor& other) const;
        [[nodiscard]]
        Tensor log() const;
        [[nodiscard]]
        Tensor log2() const;
        [[nodiscard]]
        Tensor log10() const;
        [[nodiscard]]
        Tensor exp() const;
        [[nodiscard]]
        Tensor exp2() const;
        [[nodiscard]]
        Tensor exp10() const;
        [[nodiscard]]
        Tensor pow(f32 exp = 2.0f) const;
        [[nodiscard]]
        Tensor sqrt() const;
        [[nodiscard]]
        Tensor rsqrt() const;
        [[nodiscard]]
        Tensor square() const;
        [[nodiscard]]
        Tensor sin() const;
        [[nodiscard]]
        Tensor cos() const;
        [[nodiscard]]
        Tensor tan() const;
        [[nodiscard]]
        Tensor cot() const;
        [[nodiscard]]
        Tensor abs() const;
        [[nodiscard]]
        Tensor neg() const;
        [[nodiscard]]
        Tensor sign() const;
        [[nodiscard]]
        Tensor erf() const;
        [[nodiscard]]
        Tensor inv() const;
        [[nodiscard]]
        Tensor slice(i64 dim, i64 start, i64 end) const;
        [[nodiscard]]
        Tensor clamp(f32 min, f32 max) const;
        [[nodiscard]]
        Tensor add(const Tensor& other) const;
        [[nodiscard]]
        Tensor sub(const Tensor& other) const;
        [[nodiscard]]
        Tensor mul(const Tensor& other) const;
        [[nodiscard]]
        Tensor div(const Tensor& other) const;
        [[nodiscard]]
        Tensor add(f32 value) const;
        [[nodiscard]]
        Tensor sub(f32 value) const;
        [[nodiscard]]
        Tensor mul(f32 value) const;
        [[nodiscard]]
        Tensor div(f32 value) const;

        [[nodiscard]]
        Tensor to(sys::DeviceType _device);
        [[nodiscard]]
        Tensor transpose() const;
        [[nodiscard]]
        Tensor permute(std::initializer_list<i64> dims) const;
        [[nodiscard]]
        Tensor reshape(std::initializer_list<i64> _new_shape) const;
        [[nodiscard]]
        Tensor squeeze(i64 dim = -1) const;
        [[nodiscard]]
        Tensor unsqueeze(i64 dim) const;
        [[nodiscard]]
        Tensor contiguous() const;
        [[nodiscard]]
        Tensor detach() const;
        [[nodiscard]]
        Tensor clone();

        [[nodiscard]]
        Tensor& grad();
        [[nodiscard]]
        const Tensor& grad() const;

        [[nodiscard]]
        meta::GradientPacked pack() const;

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
        friend Tensor operator+(f32 value, const Tensor& tensor);
        friend Tensor operator-(f32 value, const Tensor& tensor);
        friend Tensor operator*(f32 value, const Tensor& tensor);
        friend Tensor operator/(f32 value, const Tensor& tensor);
    private:
        std::shared_ptr<meta::GradientFlow> flow_;
        std::shared_ptr<TensorStorage> storage_;
        std::shared_ptr<Tensor> gradient_;
        TensorShape m_shape;

        bool m_require;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP