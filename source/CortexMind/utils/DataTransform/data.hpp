//
// Created by muham on 1.12.2025.
//

#ifndef CORTEXMIND_DATA_HPP
#define CORTEXMIND_DATA_HPP

#include <CortexMind/framework/Params/params.hpp>
#include <type_traits>

namespace cortex::utils {
    class TensorFactory {
        template<typename T>
        tensor from(T value, const bool requires_grad = false) {
            static_assert(std::is_arithmetic_v<T>, "Only arithmetic types are supported");

            tensor t(1, 1, 1, requires_grad);
            t(0, 0, 0) = static_cast<double>(value);
            return t;
        }
    };

    using _cast = TensorFactory;
}

#endif //CORTEXMIND_DATA_HPP