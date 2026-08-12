//
// Created by muham on 6.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_SHAPE_SHAPE_HPP
#define CORTEXMIND_FRAMEWORK_SHAPE_SHAPE_HPP

#include <CortexMind/runtime/macros.hpp>
#include <tlx/vector.hpp>
#include <vector>

namespace cortex::_fw {
    class TensorShape {
    public:
        TensorShape();
        explicit TensorShape(const tlx::vec<std::int64_t, CXM_MAX_DIMS> &shape);
        TensorShape(std::initializer_list<std::int64_t> shape);
        explicit TensorShape(const std::vector<std::int64_t>& shape);
        TensorShape(const TensorShape&);
        TensorShape(TensorShape&&) noexcept;
        ~TensorShape();

        void Set(const tlx::vec<std::int64_t, CXM_MAX_DIMS>& shape);

        [[nodiscard]]
        tlx::vec<std::int64_t, CXM_MAX_DIMS>& stride();
        [[nodiscard]]
        const tlx::vec<std::int64_t, CXM_MAX_DIMS>& stride() const;
        [[nodiscard]]
        tlx::vec<std::int64_t, CXM_MAX_DIMS>& shape();
        [[nodiscard]]
        const tlx::vec<std::int64_t, CXM_MAX_DIMS>& shape() const;

        [[nodiscard]]
        std::int64_t offset() const noexcept;

        TensorShape& operator=(const TensorShape&);
        TensorShape& operator=(TensorShape&&) noexcept;

        friend class TensorDebug;
    private:
        tlx::vec<std::int64_t, CXM_MAX_DIMS> m_shape;
        tlx::vec<std::int64_t, CXM_MAX_DIMS> m_stride;
        std::int64_t m_offset;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_SHAPE_SHAPE_HPP