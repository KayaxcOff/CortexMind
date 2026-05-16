//
// Created by muham on 16.05.2026.
//

#ifndef CORTEXMIND_TOOLS_TYPES_HPP
#define CORTEXMIND_TOOLS_TYPES_HPP

#include <CortexMind/framework/Tensor/tensor.hpp>
#include <cstdint>

namespace cortex {
    using boolean = bool;
    using u0 = void;

    using float32 = float;
    using float64 = double;

    using int8 = std::int8_t;
    using int16 = std::int16_t;
    using int32 = std::int32_t;
    using int64 = std::int64_t;
    using uint8 = std::uint8_t;
    using uint16 = std::uint16_t;
    using uint32 = std::uint32_t;
    using uint64 = std::uint64_t;

    using tensor = _fw::Tensor;
} //namespace cortex

#endif //CORTEXMIND_TOOLS_TYPES_HPP