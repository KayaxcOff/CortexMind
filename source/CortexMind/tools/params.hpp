//
// Created by muham on 4.02.2026.
//

#ifndef CORTEXMIND_TOOLS_PARAMS_HPP
#define CORTEXMIND_TOOLS_PARAMS_HPP

#include <CortexMind/core/Engine/Tensor/tensor.hpp>

namespace cortex {
    using boolean   = bool;
    using string    = std::string;
    using char8     = char;
    using int8      = int8_t;
    using int16     = int16_t;
    using int32     = int32_t;
    using int64     = int64_t;
    using float32   = float;
    using float64   = double;
    using uint8     = uint8_t;
    using uint16    = uint16_t;
    using uint32    = uint32_t;
    using uint64    = uint64_t;
    using tensor    = _fw::MindTensor;
} // namespace cortex

#endif //CORTEXMIND_TOOLS_PARAMS_HPP