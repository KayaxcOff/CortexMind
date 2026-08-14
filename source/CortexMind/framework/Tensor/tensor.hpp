//
// Created by muham on 12.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP
#define CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP

#include <CortexMind/framework/Graph/flow.hpp>
#include <CortexMind/framework/Shape/shape.hpp>
#include <CortexMind/framework/Storage/storage.hpp>
#include <CortexMind/framework/Type/trait.hpp>
#include <CortexMind/framework/Type/type.hpp>
#include <CortexMind/tools/tensor_info.hpp>
#include <tlx/concepts.hpp>
#include <tlx/span.hpp>
#include <memory>
#include <vector>

namespace cortex::_fw {
    class Tensor {
    public:
        Tensor();
        explicit Tensor(const tlx::vec<std::int64_t, CXM_MAX_DIMS>& shape, DType type = DType::Float32, DeviceType d_type = DeviceType::HOST, bool _requires_grad = false);
        explicit Tensor(const std::vector<std::int64_t>& shape, DType type = DType::Float32, DeviceType d_type = DeviceType::HOST, bool _requires_grad = false);
        explicit Tensor(const TensorInfo& info);
        ~Tensor();

        template<tlx::float_like T>
        T* get() noexcept {
            if (this->m_type.type() != ttype_of<T>) {
                return nullptr;
            }
            return this->storage_->as<T>();
        }
        template<tlx::float_like T>
        const T* get() const noexcept {
            if (this->m_type.type() != ttype_of<T>) {
                return nullptr;
            }
            return this->storage_->as<T>();
        }
        [[nodiscard]]
        bool requires_grad() const noexcept;
        [[nodiscard]]
        bool empty() const noexcept;
        [[nodiscard]]
        std::vector<std::int64_t> shape() const noexcept;
        [[nodiscard]]
        DType dtype() const noexcept;
        [[nodiscard]]
        DeviceType device() const noexcept;
        [[nodiscard]]
        std::size_t len() const noexcept;
        [[nodiscard]]
        std::size_t ndim() const noexcept;
        [[nodiscard]]
        bool has_grad() const noexcept;

        template<tlx::float_like T>
        [[nodiscard]]
        tlx::Span<T> span() noexcept {
            if (this->m_type.type() != ttype_of<T>) {
                return {};
            }
            return {this->storage_->raw(), len()};
        }
        template<tlx::float_like T>
        [[nodiscard]]
        tlx::Span<const T> span() const noexcept {
            if (this->m_type.type() != ttype_of<T>) {
                return {};
            }
            return {this->storage_->raw(), len()};
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
        friend class TensorDebug;
    private:
        std::shared_ptr<meta::GradientFlow> flow_;
        std::shared_ptr<TensorStorage> storage_;
        std::shared_ptr<Tensor> gradient_;
        TensorType m_type;
        TensorShape m_shape;
        bool m_flag;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TENSOR_TENSOR_HPP