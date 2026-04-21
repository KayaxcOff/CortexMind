//
// Created by muham on 21.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_REF_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_REF_HPP

#include <functional>

namespace cortex::_fw {
    /**
     * @brief Type alias for `std::reference_wrapper`.
     *
     * This alias provides a cleaner and more readable way to use non-owning
     * references throughout the codebase.
     *
     * `ref<T>` behaves like a reference but can be copied and stored in containers,
     * while still referring to the original object (it does not copy or own the data).
     *
     * @tparam Elem Type of the object being referenced
     */
    template<typename Elem>
    using ref = std::reference_wrapper<Elem>;
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_REF_HPP