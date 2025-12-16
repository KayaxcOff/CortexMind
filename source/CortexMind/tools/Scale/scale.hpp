//
// Created by muham on 16.12.2025.
//

#ifndef CORTEXMIND_SCALE_HPP
#define CORTEXMIND_SCALE_HPP

#include <CortexMind/framework/Params/params.hpp>
#include <CortexMind/framework/Core/AVX/funcs.hpp>

using namespace cortex::_fw;

namespace cortex::tools {
    class TensorScale {
        static tensor scale(const tensor& input, const float factor) noexcept {
            return input * factor;
        }

        static void inplaceScale(tensor& input, const float factor) noexcept {
            for (size_t i = 0; i < input.data().size(); ++i) {
                const auto v = avx2::load(&input.dataIdx(i)[0]);
                avx2::store(&input.dataIdx(i)[0], avx2::mul(v, avx2::broadcast(factor)));
            }
        }
    };
}

#endif //CORTEXMIND_SCALE_HPP