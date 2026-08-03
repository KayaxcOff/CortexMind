//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_AS_STRING_HPP
#define CORTEXMIND_AS_STRING_HPP

#include <CortexMind/framework/Type/dtype.hpp>
#include <tlx/string.hpp>

namespace cortex::_fw {
    [[nodiscard]]
    tlx::vstring as_string(DType type);
} //namespace cortex::_fw

#endif //CORTEXMIND_AS_STRING_HPP