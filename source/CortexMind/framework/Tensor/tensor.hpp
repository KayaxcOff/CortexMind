//
// Created by muham on 18.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP
#define CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP

#include <CortexMind/framework/Dispatch/txl.hpp>
#include <CortexMind/framework/Gradient/flow.hpp>
#include <CortexMind/framework/Memory/device.hpp>
#include <CortexMind/framework/Storage/stor.hpp>
#include <CortexMind/framework/Tools/params.hpp>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace cortex::_fw {

    class MindTensor {
    public:
        MindTensor();
        explicit MindTensor(const std::vector<i64>& shape, sys::deviceType device = sys::deviceType::host, bool requires_grad = false);
        MindTensor(std::initializer_list<i64> shape, sys::deviceType device = sys::deviceType::host, bool requires_grad = false);
        explicit MindTensor(const TensorStorage &tensor_storage, bool requires_grad = false);
        MindTensor(const std::vector<i64>& shape, const f32* data, sys::deviceType device = sys::deviceType::host, bool requires_grad = false);
        MindTensor(const TensorStorage& storage, const TensorStorage& grad_storage, const std::shared_ptr<meta::GradientFlow>& gradient_flow);
        MindTensor(std::shared_ptr<TensorStorage> tensor_storage, const std::shared_ptr<TensorStorage>& grad_storage, std::shared_ptr<meta::GradientFlow> gradient_flow);
        MindTensor(const MindTensor& other);
        MindTensor(MindTensor&& other) noexcept;
        ~MindTensor();

        template<typename... Args>
        requires (std::conjunction_v<std::is_integral<Args>...>)
        [[nodiscard]]
        f32& at(Args... args) {
            std::array<i64, sizeof...(Args)> idx{static_cast<i64>(args)...};
            return this->storage_->data()[compute_offset(idx)];
        }

        template<typename... Args>
        requires (std::conjunction_v<std::is_integral<Args>...>)
        [[nodiscard]]
        const f32& at(Args... args) {
            std::array<i64, sizeof...(Args)> idx{static_cast<i64>(args)...};
            return this->storage_->data()[compute_offset(idx)];
        }

        [[nodiscard]]
        f32* get();
        [[nodiscard]]
        const f32* get() const;
        [[nodiscard]]
        const std::vector<i64>& shape() const;
        [[nodiscard]]
        bool isGradRequired() const;
        [[nodiscard]]
        sys::deviceType device() const;
        [[nodiscard]]
        bool empty();

        [[nodiscard]]
        size_t len() const;
        [[nodiscard]]
        size_t ndim() const;
        [[nodiscard]]
        bool empty() const;

        [[nodiscard]]
        f32 mean() const;
        [[nodiscard]]
        f32 variance() const;
        [[nodiscard]]
        f32 stdv() const;
        [[nodiscard]]
        f32 max() const;
        [[nodiscard]]
        f32 min() const;

        void ones() const;
        void zero() const;
        void fill(f32 value) const;
        void rand(f32 min = 0.0f, f32 max = 1.0f) const;
        void backward() const;
        void backward(MindTensor& other) const;
        void set_flow(const std::shared_ptr<meta::GradientFlow>& _flow);
        void set_grad(const MindTensor& _grad);

        [[nodiscard]]
        MindTensor to(const sys::deviceType& d_type);
        [[nodiscard]]
        MindTensor dot(const MindTensor& other);
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

        [[nodiscard]]
        MindTensor& grad();
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

        txl::MatrixExecutor matrix;
        txl::ReductionOps reduction_ops;
        txl::TensorScalar scalar;
        txl::Wise wise;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP