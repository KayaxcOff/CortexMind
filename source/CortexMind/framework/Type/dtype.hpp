//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TYPE_DTYPE_HPP
#define CORTEXMIND_FRAMEWORK_TYPE_DTYPE_HPP

#include <cstdint>

namespace cortex::_fw {
    /**
     * @brief Represents the underlying scalar data type of tensors and numerical objects.
     *
     * This enumeration identifies the storage format of tensor elements and is
     * used throughout the framework for type dispatching, memory allocation,
     * serialization, and runtime type inspection.
     *
     * Both floating-point and quantized integer formats are supported to enable
     * efficient inference and training across different hardware architectures.
     */
    enum class DType : std::uint8_t {
        Unknown = 0,
        Int32 = 1,
        Int64 = 2,
        Float32 = 3,
        Float64 = 4,
        BFloat16 = 5,
        Float16 = 6,
        QInt16 = 7,
        QInt8 = 8,
        QUInt16 = 9,
        QUInt8 = 10,
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TYPE_DTYPE_HPP