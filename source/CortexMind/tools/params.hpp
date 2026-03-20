//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_TOOLS_PARAMS_HPP
#define CORTEXMIND_TOOLS_PARAMS_HPP

#include <CortexMind/core/Engine/Tensor/tensor.hpp>
#include <cstdint>

namespace cortex {
    using boolean   = bool;
    using char8     = char;
    using float32   = float;
    using float64   = double;
    using u0        = void;
    using int8      = int8_t;
    using int16     = int16_t;
    using int32     = int32_t;
    using int64     = int64_t;
    using uint8     = uint8_t;
    using uint16    = uint16_t;
    using uint32    = uint32_t;
    using uint64    = uint64_t;
    using tensor    = _fw::MindTensor;
} // namespace cortex

#endif //CORTEXMIND_TOOLS_PARAMS_HPP