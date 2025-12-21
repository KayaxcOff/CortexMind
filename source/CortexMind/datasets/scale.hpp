//
// Created by muham on 21.12.2025.
//

#ifndef CORTEXMIND_SCALE_HPP
#define CORTEXMIND_SCALE_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::ds {
    class TensorScale {
    public:
        TensorScale() = default;
        ~TensorScale() = default;

        static tensor scale(tensor& input);
    };
}

#endif //CORTEXMIND_SCALE_HPP