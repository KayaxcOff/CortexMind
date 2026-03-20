//
// Created by muham on 13.03.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

int main() {
    tensor x({2, 2});
    tensor y({2, 2});

    x.uniform();
    y.uniform();

    addition(x, y).print();

    return 0;
}
