//
// Created by muham on 30.05.2026.
//

#include "CortexMind/utility/Image/kernel.hpp"

using namespace cortex::utils;
using namespace cortex;

tensor SpatialKernel::apply(const tensor &Xx, const tensor &Xy) {
    return Xx.matmul(Xy);
}