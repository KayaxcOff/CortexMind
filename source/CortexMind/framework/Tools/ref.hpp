//
// Created by muham on 19.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_REF_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_REF_HPP

#include <type_traits>

namespace cortex::_fw {
    template<typename Elem>
    using ref = std::reference_wrapper<Elem>;
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_REF_HPP