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
        explicit MindTensor(std::shared_ptr<TensorStorage> tensor_storage);
        MindTensor(const std::vector<i64>& shape, f32* data, sys::deviceType device = sys::deviceType::host, bool requires_grad = false);
        MindTensor(MindTensor& other);
        MindTensor(MindTensor&& other) noexcept;
        ~MindTensor();

        [[nodiscard]]
        f32* get();
        [[nodiscard]]
        const f32* get() const;
        [[nodiscard]]
        const std::vector<i64>& shape();
        [[nodiscard]]
        bool requires_grad();
    private:
        std::shared_ptr<meta::GradientFlow> flow_;
        std::shared_ptr<TensorStorage> storage_;
        std::unique_ptr<MindTensor> gradient_;

        sys::deviceType m_device;

        bool m_grad_flag;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP