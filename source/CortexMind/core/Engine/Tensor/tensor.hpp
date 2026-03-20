//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP
#define CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP

#include <CortexMind/core/Engine/Memory/device.hpp>
#include <CortexMind/core/Engine/Storage/stor.hpp>
#include <CortexMind/core/Tools/err.hpp>
#include <CortexMind/core/Tools/params.hpp>
#include <CortexMind/core/Tools/tensor_utils.hpp>
#include <CortexMind/core/Graph/flow.hpp>
#include <initializer_list>
#include <memory>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief   Multidimensional tensor with device abstraction and basic math operations
     */
    class MindTensor {
    public:
        /**
         * @brief   Default constructor – creates empty tensor (size=0, device=host)
         */
        MindTensor();
        /**
         * @brief   Constructs tensor with given shape on specified device
         * @param   shape           Dimensions (e.g. {batch, channels, height, width})
         * @param   d               Target device (host or cuda)
         * @param requires_grad     Active grad
         */
        explicit MindTensor(const std::vector<i64>& shape, sys::dev d = sys::dev::host, bool requires_grad = false);
        /**
         * @brief   Constructs tensor from initializer list (convenience overload)
         */
        MindTensor(const std::initializer_list<i64>& shape, sys::dev d = sys::dev::host, bool requires_grad = false);
        /**
         * @brief   Constructs tensor from existing data buffer
         * @param   shape           Dimensions
         * @param   data            Raw data pointer (host or device)
         * @param   d               Device of the data pointer
         * @param requires_grad     Active grad
         *
         * @note    Copies data into new storage (deep copy)
         */
        MindTensor(const std::vector<i64>& shape, const f32* data, sys::dev d = sys::dev::host, bool requires_grad = false);
        MindTensor(const MindTensor& other);
        MindTensor(MindTensor&& other) noexcept;
        ~MindTensor();

        /**
         * @brief   Multidimensional indexing (host-only)
         * @tparam  Args    Integer index types (i64, int, size_t, etc.)
         * @param   args    Indices in each dimension
         * @return  Reference to element at position
         *
         * @pre     Tensor must be on host (dev::host)
         * @pre     Number of indices == ndim()
         * @pre     All indices in valid range [0, shape[d])
         */
        template<typename ... Args>
        [[nodiscard]]
        f32& at(Args ... args) {
            CXM_ASSERT(this->m_dev == sys::dev::host, "cortex::_fw::MindTensor::at()", "Tensor must be on HOST Device for index access");
            static_assert((std::is_integral_v<Args> && ...), "`at()` only retrieves the integer index.");
            CXM_ASSERT(this->storage_->isValid(), "cortex::_fw::MindTensor::at()", "Storage is null.");
            const std::vector<i64> indexVec = {static_cast<i64>(args)...};
            CXM_ASSERT(static_cast<i64>(indexVec.size()) == this->ndim(), "cortex::_fw::MindTensor::at()", "The index dimension does not match the tensor dimension.");
            for (i64 i = 0; i < static_cast<i64>(indexVec.size()); ++i) {
                CXM_ASSERT(indexVec[i] >= 0 && indexVec[i] < this->m_shape[i], "cortex::_fw::MindTensor::at()", "Index delimited.");
            }
            const i64 idx = compute_offset(indexVec, this->m_stride);
            return this->storage_->data()[idx];
        }
        /**
         * @brief   Multidimensional indexing (host-only)
         * @tparam  Args    Integer index types (i64, int, size_t, etc.)
         * @param   args    Indices in each dimension
         * @return  Const reference to element at position
         *
         * @pre     Tensor must be on host (dev::host)
         * @pre     Number of indices == ndim()
         * @pre     All indices in valid range [0, shape[d])
         */
        template<typename ... Args>
        [[nodiscard]]
        const f32& at(Args ... args) const {
            CXM_ASSERT(this->m_dev == sys::dev::host, "cortex::_fw::MindTensor::at()", "Tensor must be on HOST Device for index access");
            static_assert((std::is_integral_v<Args> && ...), "`at()` only retrieves the integer index.");
            CXM_ASSERT(this->storage_->isValid(), "cortex::_fw::MindTensor::at()", "Storage is null.");
            const std::vector<i64> indexVec = {static_cast<i64>(args)...};
            CXM_ASSERT(static_cast<i64>(indexVec.size()) == this->ndim(), "cortex::_fw::MindTensor::at()", "The index dimension does not match the tensor dimension.");
            for (i64 i = 0; i < static_cast<i64>(indexVec.size()); ++i) {
                CXM_ASSERT(indexVec[i] >= 0 && indexVec[i] < this->m_shape[i], "cortex::_fw::MindTensor::at()", "Index delimited.");
            }
            const i64 idx = compute_offset(indexVec, this->m_stride);
            return this->storage_->data()[idx];
        }

        /**
         * @brief   Returns raw data pointer (host or device depending on .device())
         * @return  float* pointing to first element (offset applied)
         *
         * @warning Do not use on GPU tensors for direct indexing
         */
        [[nodiscard]]
        f32* get();
        /**
         * @brief   Returns const raw data pointer (host or device depending on .device())
         * @return  f32* pointing to first element (offset applied)
         *
         * @warning Do not use on GPU tensors for direct indexing
         */
        [[nodiscard]]
        f32* get() const;
        /**
         * @brief   Returns current shape vector
         */
        [[nodiscard]]
        const std::vector<i64>& shape() const;
        /**
         * @brief   Returns current stride vector (row-major)
         */
        [[nodiscard]]
        const std::vector<i64>& stride() const;
        /**
         * @brief   Returns total number of elements (product of shape)
         */
        [[nodiscard]]
        size_t numel() const noexcept;
        /**
         * @brief   Returns number of dimensions (rank)
         */
        [[nodiscard]]
        i64 ndim() const noexcept;
        [[nodiscard]]
        bool grad_required() const noexcept;
        /**
         * @brief   Returns true if tensor has zero elements
         */
        [[nodiscard]]
        bool empty() const noexcept;
        /**
         * @brief   Returns true if memory layout is contiguous (stride matches shape)
         */
        [[nodiscard]]
        bool contiguous() const noexcept;
        /**
         * @brief   Returns current device (host or cuda)
         */
        [[nodiscard]]
        sys::dev device() const noexcept;

        /**
         * @brief   Computes mean of all elements
         * @return  Arithmetic mean (sum / numel)
         */
        [[nodiscard]]
        f32 mean() const;
        /**
         * @brief   Computes variance of all elements
         * @return  Variance (sum((x - mean)²) / numel)
         */
        [[nodiscard]]
        f32 variance() const;
        /**
         * @brief   Computes standard deviation
         * @return  std = √variance
         */
        [[nodiscard]]
        f32 std_dev() const;
        /**
         * @brief   Computes maximum value
         */
        [[nodiscard]]
        f32 max() const;
        /**
         * @brief   Computes minimum value
         */
        [[nodiscard]]
        f32 min() const;
        /**
         * @brief   Computes Euclidean (L2) norm √(sum(x²))
         */
        [[nodiscard]]
        f32 norm() const;
        /**
         * @brief   Computes sum of all elements
         */
        [[nodiscard]]
        f32 sum_all() const;

        void backward() const;
        void backward(MindTensor* _grad) const;
        void print() const;
        /**
         * @brief   Fills tensor with uniform random values in [min, max]
         * @param   min     Lower bound (default 0)
         * @param   max     Upper bound (default 1)
         */
        void uniform(f32 min = 0.0f, f32 max = 1.0f);
        /**
         * @brief   Fills tensor with zeros
         */
        void zero() const;
        /**
         * @brief   Fills tensor with ones
         */
        void ones() const;
        /**
         * @brief   Fills tensor with constant value
         * @param   val     Fill value
         */
        void fill(f32 val) const;
        void require_grad(bool _require_grad);
        void set_grad(std::unique_ptr<MindTensor> _grad);
        void set_grad(const MindTensor& grad);
        void set_flow(std::shared_ptr<meta::GradientFlow> _flow);

        /**
         * @brief   Moves tensor to target device (copies data if needed)
         * @param   _device   Target device (host or cuda)
         * @return  Reference to self (for chaining)
         */
        [[nodiscard]]
        MindTensor to(sys::dev _device);
        /**
         * @brief   Returns flattened view (2D: batch × features)
         * @note    Requires contiguous memory
         */
        [[nodiscard]]
        MindTensor flat() const;
        /**
         * @brief   Matrix multiplication (2D only)
         * @param   other   Right-hand matrix
         * @return  Result tensor (new allocation)
         */
        [[nodiscard]]
        MindTensor matmul(MindTensor other);
        /**
         * @brief   Permutes dimensions according to axes order
         * @param   axes    New dimension order
         * @return  View tensor (shared storage, updated stride)
         */
        [[nodiscard]]
        MindTensor permute(const std::vector<i64>& axes) const;
        /**
         * @brief   Creates deep copy of tensor (new storage)
         */
        [[nodiscard]]
        MindTensor copy() const;
        /**
         * @brief   Returns tensor without gradient tracking/flow
         */
        [[nodiscard]]
        MindTensor detach() const;
        /**
         * @brief   Reshapes tensor (must preserve total elements)
         * @note    Requires contiguous memory
         */
        [[nodiscard]]
        MindTensor reshape(const std::vector<i64>& shape) const;
        /**
         * @brief   Element-wise square root
         */
        [[nodiscard]]
        MindTensor sqrt();
        /**
         * @brief   Element-wise power (x^value)
         * @param   value   Exponent (default 2 → square)
         */
        [[nodiscard]]
        MindTensor pow(f32 value = 2.0f);
        /**
         * @brief   Sum of all elements (scalar result, 1D shape)
         */
        [[nodiscard]]
        MindTensor sum();
        /**
         * @brief   Sum reduction along given dimension
         * @param   dim     Dimension to reduce
         * @param   keep    Keep reduced dimension as size 1 (default false)
         */
        [[nodiscard]]
        MindTensor sum(i64 dim, bool keep = false);
        /**
         * @brief   Transposes 2D tensor (swaps dimensions 0 and 1)
         */
        [[nodiscard]]
        MindTensor transpose();
        /**
         * @brief   Element-wise exponential (exp(x))
         */
        [[nodiscard]]
        MindTensor exp();
        /**
         * @brief   Element-wise natural logarithm (log(x))
         */
        [[nodiscard]]
        MindTensor log();
        /**
         * @brief   Element-wise absolute value (|x|)
         */
        [[nodiscard]]
        MindTensor abs();
        /**
         * @brief   Adds new dimension of size 1 at given position
         * @param   dim     Insertion position (-ndim to ndim)
         */
        [[nodiscard]]
        MindTensor unsqueeze(i64 dim) const;
        /**
         * @brief   Removes dimension of size 1 at given position
         * @param   dim     Dimension to remove
         */
        [[nodiscard]]
        MindTensor squeeze(i64 dim) const;
        /**
         * @brief   Slices along dimension 0 (first axis)
         * @param   start   Start index (inclusive)
         * @param   end     End index (exclusive)
         */
        [[nodiscard]]
        MindTensor slice(i64 start, i64 end) const;

        [[nodiscard]]
        MindTensor& grad();
        [[nodiscard]]
        const MindTensor& grad() const;

        MindTensor operator+(const MindTensor& other) ;
        MindTensor operator-(const MindTensor& other);
        MindTensor operator*(const MindTensor& other);
        MindTensor operator/(const MindTensor& other);

        MindTensor& operator+=(const MindTensor& other);
        MindTensor& operator-=(const MindTensor& other);
        MindTensor& operator*=(const MindTensor& other);
        MindTensor& operator/=(const MindTensor& other);

        MindTensor operator+(f32 value);
        MindTensor operator-(f32 value);
        MindTensor operator*(f32 value);
        MindTensor operator/(f32 value);

        MindTensor& operator+=(f32 value);
        MindTensor& operator-=(f32 value);
        MindTensor& operator*=(f32 value);
        MindTensor& operator/=(f32 value);

        friend
        MindTensor operator+(f32 scalar, MindTensor& t);
        friend
        MindTensor operator*(f32 scalar, MindTensor& t);

        bool operator==(const MindTensor& other) const;
        bool operator!=(const MindTensor& other) const;

        MindTensor& operator=(const MindTensor& other);
        MindTensor& operator=(MindTensor&& other) noexcept;
    private:
        std::shared_ptr<TensorStorage> storage_;
        std::shared_ptr<meta::GradientFlow> flow_;
        std::unique_ptr<MindTensor> gradient_;

        sys::dev m_dev;

        std::vector<i64> m_shape;
        std::vector<i64> m_stride;
        i64 m_offset;
        bool m_grad_flag;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_ENGINE_TENSOR_TENSOR_HPP