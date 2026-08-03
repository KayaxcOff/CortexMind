//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TYPE_SIZE_HPP
#define CORTEXMIND_FRAMEWORK_TYPE_SIZE_HPP

#include <CortexMind/framework/Type/dtype.hpp>
#include <cwchar>

namespace cortex::_fw {
    /**
     * @brief Returns the storage size, in bytes, of a CortexMind data type.
     *
     * Determines the size of the scalar type represented by the specified
     * @ref DType value. This function is commonly used for memory allocation,
     * tensor layout computation, serialization, and runtime type inspection.
     *
     * If the specified data type is @ref DType::Unknown, the function returns
     * zero.
     *
     * @param type The data type whose storage size is requested.
     *
     * @return The size of the corresponding scalar type in bytes, or
     *         `0` if the data type is unknown.
     *
     * @note The returned size corresponds to a single scalar element,
     * not the size of an entire tensor.
     */
    [[nodiscard]]
    std::size_t sizeOf(DType type) noexcept;
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TYPE_SIZE_HPP