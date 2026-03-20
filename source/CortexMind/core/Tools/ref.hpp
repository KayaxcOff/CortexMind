//
// Created by muham on 17.03.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_REF_HPP
#define CORTEXMIND_CORE_TOOLS_REF_HPP

#include <type_traits>

namespace cortex::_fw {
    /**
     * @brief wrapper for reference_wrapper
     */
    template <class Elem>
    using ref = std::reference_wrapper<Elem>;
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_TOOLS_REF_HPP