//
// Created by muham on 4.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP
#define CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP

#include <CortexMind/core/Engine/Storage/stor.hpp>
#include <CortexMind/core/Graph/Flow/flow.hpp>
#include <CortexMind/core/Tools/utils.hpp>
#include <initializer_list>
#include <memory>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief Multidimensional tensor with automatic differentiation support.
     *
     * MindTensor represents a contiguous or striped view over tensor data,
     * backed by TensorStorage. It supports basic arithmetic operations,
     * shape manipulation, and automatic differentiation through a
     * gradient flow graph.
     *
     * A tensor may:
     *  - own its storage
     *  - share storage with other tensors (views)
     *  - participate in a computation graph if it requires_grad is enabled
     */
    class MindTensor {
    public:
        /**
         * @brief Constructs a tensor with the given shape.
         * @param shape Tensor dimensions.
         * @param _requires_grad Enables gradient tracking.
         */
        explicit MindTensor(std::vector<i64> shape, bool _requires_grad = false);

        /// Constructs an empty tensor.
        MindTensor();
        /**
         * @brief Constructs a tensor from an initializer list shape.
         */
        MindTensor(std::initializer_list<i64> shape, bool _requires_grad = false);
        /// Copy constructor (shares storage, copies metadata).
        MindTensor(const MindTensor& other);
        /// Default destructor.
        ~MindTensor() = default;

        template<typename... Args>
        [[nodiscard]]
        f32& at(Args... indices) {
            const std::vector<i64> vecIdx{static_cast<i64>(indices)...};
            const i64 idx = compute_offset(vecIdx, this->m_strides);
            return this->storage_->data()[idx];
        }

        template<typename... Args>
        [[nodiscard]]
        const f32& at(Args... indices) const {
            const std::vector<i64> vecIdx{static_cast<i64>(indices)...};
            const i64 idx = compute_offset(vecIdx, this->m_strides);
            return this->storage_->data()[idx];
        }

        [[nodiscard]]
        f32& at(const std::vector<i64>& indices);
        [[nodiscard]]
        const f32& at(const std::vector<i64>& indices) const;
        [[nodiscard]]
        f32* get();
        [[nodiscard]]
        const f32* get() const;
        [[nodiscard]]
        std::vector<i64> shape();
        [[nodiscard]]
        size_t numel() const;
        [[nodiscard]]
        bool empty() const;
        [[nodiscard]]
        bool requires_grad() const;
        [[nodiscard]]
        bool is_contiguous() const;
        [[nodiscard]]
        f32 mean() const;
        [[nodiscard]]
        f32 variance() const;
        [[nodiscard]]
        f32 max() const;
        [[nodiscard]]
        f32 min() const;

        void backward();
        void print() const;
        void uniform_rand(f32 min = 0.0, f32 max = 1.0) const;
        void zero() const;
        void ones() const;
        void fill(f32 value) const;
        void require_grad(bool _require_grad);
        void set_grad(std::unique_ptr<MindTensor> _grad);
        void set_grad(const MindTensor& grad);
        void zero_grad() const;

        [[nodiscard]]
        MindTensor flatten() const;
        [[nodiscard]]
        MindTensor matmul(const MindTensor& other);
        [[nodiscard]]
        MindTensor permute(const std::vector<i64>& axes) const;
        [[nodiscard]]
        MindTensor copy() const;
        [[nodiscard]]
        MindTensor clone() const;
        [[nodiscard]]
        MindTensor reshape(const std::vector<i64>& shape) const;
        [[nodiscard]]
        MindTensor sqrt() const;
        [[nodiscard]]
        MindTensor pow(f32 value = 2.0f) const;
        [[nodiscard]]
        MindTensor sum() const;
        [[nodiscard]]
        MindTensor sum(i64 dim, bool keep = false) const;
        [[nodiscard]]
        MindTensor expand(const std::vector<i64>& shape) const;
        [[nodiscard]]
        MindTensor transpose();
        [[nodiscard]]
        MindTensor exp() const;
        [[nodiscard]]
        MindTensor log() const;
        [[nodiscard]]
        MindTensor abs() const;
        [[nodiscard]]
        MindTensor unsqueeze(i64 dim) const;
        [[nodiscard]]
        MindTensor squeeze(i64 dim) const;

        [[nodiscard]]
        MindTensor& grad();
        [[nodiscard]]
        const MindTensor& grad() const;

        MindTensor operator+(const MindTensor& other) const;
        MindTensor operator-(const MindTensor& other) const;
        MindTensor operator*(const MindTensor& other) const;
        MindTensor operator/(const MindTensor& other) const;

        MindTensor& operator+=(const MindTensor& other);
        MindTensor& operator-=(const MindTensor& other);
        MindTensor& operator*=(const MindTensor& other);
        MindTensor& operator/=(const MindTensor& other);

        MindTensor operator+(f32 scalar) const;
        MindTensor operator-(f32 scalar) const;
        MindTensor operator*(f32 scalar) const;
        MindTensor operator/(float scalar) const;

        MindTensor& operator+=(f32 scalar);
        MindTensor& operator-=(f32 scalar);
        MindTensor& operator*=(f32 scalar);
        MindTensor& operator/=(f32 scalar);

        bool operator==(const MindTensor& other) const;
        bool operator!=(const MindTensor& other) const;
    private:
        std::shared_ptr<TensorStorage> storage_;
        std::unique_ptr<MindTensor> gradient_;
        std::shared_ptr<meta::GradientFlow> flow_;

        std::vector<i64> m_shape;
        std::vector<i64> m_strides;
        i64 m_offset;
        bool m_grad_flag;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP