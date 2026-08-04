//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_NATIVE_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_NATIVE_HPP

namespace cortex::_fw {
    /**
     * @brief Stores two values of the same native type.
     *
     * native is a lightweight aggregate used to group two scalar values
     * into a single object. The intended semantics of the two components
     * depend on the context in which the structure is used.
     *
     * @tparam T Underlying scalar type.
     */
    template<typename T>
    struct native {
        T low;
        T high;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_NATIVE_HPP