//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TYPE_TYPE_HPP
#define CORTEXMIND_FRAMEWORK_TYPE_TYPE_HPP

#include <CortexMind/framework/Type/dtype.hpp>
#include <string_view>

namespace cortex::_fw {
    /**
     * @brief Represents the runtime data type of tensor.
     *
     * TensorType is a lightweight wrapper around @ref DType that provides
     * convenient utility functions for runtime type inspection, string
     * conversion, and future type-related operations.
     *
     * Although internally it stores only a @ref DType value, encapsulating
     * the data type within a dedicated class allows the framework to extend
     * its functionality without changing the tensor interface.
     *
     * Typical use cases include debugging, serialization, logging, runtime
     * validation, and memory layout inspection.
     */
    class TensorType {
    public:
        /**
         * @brief Constructs a tensor type from the specified data type.
         *
         * @param type The underlying CortexMind data type.
         */
        explicit TensorType(DType type);
        TensorType(const TensorType&);
        TensorType(TensorType&&) noexcept;
        ~TensorType();

        /**
         * @brief Returns the underlying data type.
         *
         * @return The stored @ref DType value.
         */
        [[nodiscard]]
        DType type() const noexcept;
        /**
         * @brief Returns the canonical textual representation of the stored data type.
         *
         * @return A string view containing the canonical type name.
         */
        [[nodiscard]]
        std::string_view ToString() const noexcept;

        TensorType& operator=(const TensorType&);
        TensorType& operator=(TensorType&&) noexcept;
    private:
        DType m_type;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TYPE_TYPE_HPP