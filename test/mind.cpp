//
// Created by muham on 30.11.2025.
//

#include <CortexMind/framework/Params/params.hpp>
#include <CortexMind/utils/MathTools/pch.hpp>

using namespace cortex;

int main() {
    tensor x{2, 3, 3};

    tensor y{2, 3, 3};

    x.uniform_rand(-1.0, 1.0);
    y.uniform_rand(-1.0, 1.0);

    add(x, y).print();

    return 0;
}