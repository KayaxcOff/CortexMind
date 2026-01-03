//
// Created by muham on 29.12.2025.
//

#ifndef CORTEXMIND_CORE_PARAMS_PARAMS_HPP
#define CORTEXMIND_CORE_PARAMS_PARAMS_HPP

#include <core/Engine/Tensor/tensor.hpp>

#include <cstddef>

namespace cortex {
    using size = std::size_t;
    using int32 = int;
    using int64 = long long;
    using float32 = float;
    using float64 = double;
    using uint8 = unsigned char;
    using uint16 = unsigned short;
    using uint32 = unsigned int;
    using uint64 = unsigned long long;
    using tensor = _fw::MindTensor;
} // namespace cortex

#endif // CORTEXMIND_CORE_PARAMS_PARAMS_HPP