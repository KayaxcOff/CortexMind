//
// Created by muham on 29.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_DTYPE_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_DTYPE_HPP

namespace cortex::_fw {
    /**
     * @brief Supported data types for the Series class and related utilities.
     *
     * Used to provide type safety and runtime type information in variant-based
     * containers.
     */
    enum class DType {
        Float32,   ///< 32-bit floating point numbers (`f32`)
        Bool,      ///< Boolean values (`bool`)
        String     ///< String values (`std::string`)
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_DTYPE_HPP