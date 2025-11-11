//
// Created by muham on 8.11.2025.
//

#ifndef CORTEXMIND_PARAMS_HPP
#define CORTEXMIND_PARAMS_HPP

#include <cstddef>
#include <CortexMind/Utils/MathTools/Object/tensor.hpp>
#include <CortexMind/Utils/MathTools/Object/vector.hpp>

namespace cortex {
    using int32 = int;
    using float32 = float;
    using float64 = double;
    using size = std::size_t;
    using tensor = vecs::MindTensor;
    using matrix = vecs::MindVector;
}

#endif //CORTEXMIND_PARAMS_HPP