//
// Created by muham on 22.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP
#define CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP

#include <CortexMind/core/Engine/Storage/stor.hpp>
#include <CortexMind/core/Engine/AVX2/params.hpp>
#include <CortexMind/core/Engine/Memory/buffer.hpp>
#include <CortexMind/core/Graph/flow.hpp>
#include <CortexMind/core/Tools/params.hpp>
#include <CortexMind/core/Tools/tensor_utils.hpp>
#include <CortexMind/core/Tools/err.hpp>
#include <initializer_list>
#include <memory>
#include <vector>

namespace cortex::_fw {
    class MindTensor {
    public:
        MindTensor();

        explicit
        MindTensor(const std::vector<i64>& shape, bool requires_grad = false);
        MindTensor(std::initializer_list<i64> shape, bool requires_grad = false);
        MindTensor(const std::vector<i64>& shape, const f32* data, bool requires_grad = false);
        MindTensor(const std::vector<i64>& shape, sys::device _device, bool requires_grad = false);
        MindTensor(const MindTensor& other);
        MindTensor(MindTensor&& other) noexcept;
        ~MindTensor();

        template<typename ... Args>
        [[nodiscard]]
        f32& at(Args... indices) {
            static_assert((std::is_integral_v<Args> && ...), "`at()` only retrieves the integer index.");
            CXM_ASSERT(this->storage_->isValid(), "cortex::_fw::MindTensor::at()", "Storage is null.");
            CXM_ASSERT(this->storage_->is_device(sys::device::host), "cortex::_fw::MindTensor::at()", "CPU access to the GPU tensor is not possible.");
            const std::vector<i64> indexVec = {static_cast<i64>(indices)...};
            CXM_ASSERT(static_cast<i64>(indexVec.size()) == this->ndim(), "cortex::_fw::MindTensor::at()", "The index dimension does not match the tensor dimension.");
            for (i64 i = 0; i < static_cast<i64>(indexVec.size()); ++i) {
                CXM_ASSERT(indexVec[i] >= 0 && indexVec[i] < this->m_shape[i], "cortex::_fw::MindTensor::at()", "Index delimited.");
            }
            const i64 idx = compute_offset(indexVec, this->m_stride);
            return this->storage_->data()[idx];
        }
        template<typename ... Args>
        [[nodiscard]]
        const f32& at(Args... indices) const {
            static_assert((std::is_integral_v<Args> && ...), "`at()` only retrieves the integer index.");
            CXM_ASSERT(this->storage_->isValid(), "cortex::_fw::MindTensor::at()", "Storage is null.");
            CXM_ASSERT(this->storage_->is_device(sys::device::host), "cortex::_fw::MindTensor::at()", "CPU access to the GPU tensor is not possible.");
            const std::vector<i64> indexVec = {static_cast<i64>(indices)...};
            CXM_ASSERT(static_cast<i64>(indexVec.size()) == this->ndim(), "cortex::_fw::MindTensor::at()", "The index dimension does not match the tensor dimension.");
            for (i64 i = 0; i < static_cast<i64>(indexVec.size()); ++i) {
                CXM_ASSERT(indexVec[i] >= 0 && indexVec[i] < this->m_shape[i], "cortex::_fw::MindTensor::at()", "Index delimited.");
            }
            const i64 idx = compute_offset(indexVec, this->m_stride);
            return this->storage_->data()[idx];
        }

        [[nodiscard]]
        f32& at(const std::vector<i64>& indices);
        [[nodiscard]]
        const f32& at(const std::vector<i64>& indices) const;
        [[nodiscard]]
        f32* get();
        [[nodiscard]]
        f32* get() const;
        [[nodiscard]]
        sys::buffer* buffer();
        [[nodiscard]]
        sys::buffer* buffer() const;
        [[nodiscard]]
        std::vector<i64> shape() const noexcept;
        [[nodiscard]]
        i64 ndim() const noexcept;
        [[nodiscard]]
        bool requires_grad() const noexcept;
        [[nodiscard]]
        bool has_grad() const noexcept;
        [[nodiscard]]
        bool is_contiguous() const noexcept;
        [[nodiscard]]
        bool empty() const noexcept;
        [[nodiscard]]
        sys::device devices() const noexcept;

        [[nodiscard]]
        f32 mean() const;
        [[nodiscard]]
        f32 variance() const;
        [[nodiscard]]
        f32 std_dev() const;
        [[nodiscard]]
        f32 max() const;
        [[nodiscard]]
        f32 min() const;
        [[nodiscard]]
        f32 norm() const;
        [[nodiscard]]
        f32 sum_all() const;
        [[nodiscard]]
        size_t numel() const noexcept;

        void backward() const;
        void keep_backward(MindTensor* _grad) const;
        void print() const;
        void print_shape() const;
        void print_stride() const;
        void uniform_rand(f32 min = 0.0f, f32 max = 1.0f);
        void zero() const;
        void ones() const;
        void fill(f32 val) const;
        void require_grad(bool _require_grad);
        void set_grad(std::unique_ptr<MindTensor> _grad);
        void set_grad(const MindTensor& grad);
        void zero_grad() const;
        void set_flow(std::shared_ptr<meta::GradientFlow> _flow);
        void clear_flow();

        [[nodiscard]]
        MindTensor to(sys::device _device);
        [[nodiscard]]
        MindTensor flatten() const;
        [[nodiscard]]
        MindTensor matmul(const MindTensor& other);
        [[nodiscard]]
        MindTensor permute(const std::vector<i64>& axes) const;
        [[nodiscard]]
        MindTensor clone() const;
        [[nodiscard]]
        MindTensor detach() const;
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
        MindTensor repeat(i64 times, i64 dim);
        [[nodiscard]]
        MindTensor transpose();
        [[nodiscard]]
        MindTensor exp() const;
        [[nodiscard]]
        MindTensor log() const;
        [[nodiscard]]
        MindTensor abs() const;
        [[nodiscard]]
        MindTensor relu();
        [[nodiscard]]
        MindTensor tanh() const;
        [[nodiscard]]
        MindTensor sigmoid();
        [[nodiscard]]
        MindTensor unsqueeze(i64 dim) const;
        [[nodiscard]]
        MindTensor squeeze(i64 dim) const;
        [[nodiscard]]
        MindTensor eq(const MindTensor& other) const;
        [[nodiscard]]
        MindTensor ne(const MindTensor& other) const;
        [[nodiscard]]
        MindTensor gt(const MindTensor& other) const;
        [[nodiscard]]
        MindTensor lt(const MindTensor& other) const;
        [[nodiscard]]
        MindTensor slice(i64 start, i64 end) const;

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
        MindTensor operator/(f32 scalar) const;

        MindTensor& operator+=(f32 scalar);
        MindTensor& operator-=(f32 scalar);
        MindTensor& operator*=(f32 scalar);
        MindTensor& operator/=(f32 scalar);

        friend
        MindTensor operator+(f32 scalar, const MindTensor& t);
        friend
        MindTensor operator*(f32 scalar, const MindTensor& t);

        bool operator==(const MindTensor& other) const;
        bool operator!=(const MindTensor& other) const;

        MindTensor& operator=(const MindTensor& other);
        MindTensor& operator=(MindTensor&& other) noexcept;
    private:
        std::shared_ptr<TensorStorage> storage_;
        std::shared_ptr<meta::GradientFlow> flow_;
        std::unique_ptr<MindTensor> gradient_;

        sys::device m_device;

        std::vector<i64> m_stride;
        std::vector<i64> m_shape;
        i64 m_offset;
        bool m_require;

        [[nodiscard]]
        f32 gpu_reduce(const std::string& partial_kernel, const std::string& final_kernel, cl_float identity) const;
        static MindTensor elementwise_cmp(const MindTensor& a, const MindTensor& b, const char* name, avx2::vec8f (*avx2_op)(const avx2::vec8f&, const avx2::vec8f&), bool (*scalar_op)(f32, f32), void (*cl_op)(const sys::buffer&, const sys::buffer&, sys::buffer&));
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP