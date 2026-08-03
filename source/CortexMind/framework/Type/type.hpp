//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TYPE_TYPE_HPP
#define CORTEXMIND_FRAMEWORK_TYPE_TYPE_HPP

#include <CortexMind/framework/Type/dtype.hpp>
#include <string_view>

namespace cortex::_fw {
    class TensorType {
    public:
        explicit TensorType(DType type);
        TensorType(const TensorType&);
        TensorType(TensorType&&) noexcept;
        ~TensorType();

        [[nodiscard]]
        DType type() const noexcept;
        [[nodiscard]]
        std::string_view ToString() const noexcept;

        TensorType& operator=(const TensorType&);
        TensorType& operator=(TensorType&&) noexcept;
    private:
        DType m_type;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TYPE_TYPE_HPP