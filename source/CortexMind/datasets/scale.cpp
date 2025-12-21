//
// Created by muham on 21.12.2025.
//

#include "CortexMind/datasets/scale.hpp"
#include <CortexMind/framework/Tools/Debug/catch.hpp>
#include <CortexMind/framework/Core/AVX/funcs.hpp>
#include <iostream>

using namespace cortex::ds;
using namespace cortex::_fw;
using namespace cortex;

tensor TensorScale::scale(tensor& input) {
    if (input.width() == 0 || input.height() == 0) CXM_ASSERT(true, "Dataset is empty");

    for (auto& item : input.data()) {
        const avx2::reg va = avx2::load(item.ptr());
        const avx2::reg vb = avx2::broadcast(255.0f);
        const avx2::reg vc = avx2::div(va, vb);
        avx2::store(item.ptr(), vc);
    }
    return input;
}
