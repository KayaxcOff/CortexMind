//
// Created by muham on 25.02.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_REF_HPP
#define CORTEXMIND_CORE_TOOLS_REF_HPP

#include <functional>

namespace cortex::_fw {
    template <class _Elem>
    using ref = std::reference_wrapper<_Elem>;
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_TOOLS_REF_HPP