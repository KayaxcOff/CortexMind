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
    class MindTensor {
    public:
        MindTensor();
        explicit MindTensor(const std::vector<i64>& shape, sys::deviceType device = sys::deviceType::host, bool requires_grad = false);
        MindTensor(std::initializer_list<i64> shape, sys::deviceType device = sys::deviceType::host, bool requires_grad = false);
        explicit MindTensor(const std::shared_ptr<TensorStorage> &tensor_storage, bool requires_grad = false);
        MindTensor(const std::vector<i64>& shape, const f32* data, sys::deviceType device = sys::deviceType::host, bool requires_grad = false);
        MindTensor(const MindTensor& other);
        MindTensor(MindTensor&& other) noexcept;
        ~MindTensor();

        [[nodiscard]]
        f32* get();
        [[nodiscard]]
        const f32* get() const;
        [[nodiscard]]
        const std::vector<i64>& shape() const;
        [[nodiscard]]
        bool requires_grad() const;
        [[nodiscard]]
        sys::deviceType device() const;

        [[nodiscard]]
        size_t size() const;

        [[nodiscard]]
        f32 mean() const;
        [[nodiscard]]
        f32 variance() const;
        [[nodiscard]]
        f32 standard_deviation() const;
        [[nodiscard]]
        f32 max() const;
        [[nodiscard]]
        f32 min() const;

        void ones() const;
        void zero() const;
        void fill(f32 value) const;
        void rand(f32 min = 0.0f, f32 max = 1.0f);
        void backward();
        void backward(MindTensor& other);

        [[nodiscard]]
        MindTensor flat();
        [[nodiscard]]
        MindTensor dot(MindTensor other);
        [[nodiscard]]
        MindTensor pow(f32 exp = 2);
        [[nodiscard]]
        MindTensor square();
    private:
        std::shared_ptr<meta::GradientFlow> flow_;
        std::shared_ptr<TensorStorage> storage_;
        std::unique_ptr<MindTensor> gradient_;

        bool m_grad_flag;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP