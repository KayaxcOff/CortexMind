//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_AS_STRING_HPP
#define CORTEXMIND_AS_STRING_HPP

#include <CortexMind/framework/Type/dtype.hpp>
#include <tlx/string.hpp>

namespace cortex::_fw {
    /**
     * @brief Converts a data type identifier to its human-readable string representation.
     *
     * Returns the canonical textual name associated with a given
     * @ref DType value. The returned string is primarily intended
     * for logging, debugging, serialization, and diagnostic messages.
     *
     * If the specified data type is unknown or unsupported,
     * the function returns `"unknown"`.
     *
     * @param type The data type to convert.
     *
     * @return A string containing the canonical name of the specified data type.
     *
     * @note The returned names follow the CortexMind naming convention
     * (e.g. `"float32"`, `"int64"`, `"bfloat16"`).
     */
    [[nodiscard]]
    tlx::vstring as_string(DType type);
} //namespace cortex::_fw

#endif //CORTEXMIND_AS_STRING_HPP